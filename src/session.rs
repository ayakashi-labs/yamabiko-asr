use crate::backend::{AsrModel, BackendTranscript, RecognitionStream};
use crate::event::{SegmentId, TranscriptEvent, TranscriptSegment};
use crate::vad::{SpeechChunk, VadFactory, VadGate, duration_from_samples};
use crate::{AudioSourceConfig, AudioSourceId, Error, PcmChunk, Result, TranscriberConfig};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Join handle for the blocking transcription worker.
pub type TranscriptionWorker = JoinHandle<()>;

/// Input handle for one registered audio source.
///
/// The handle is intentionally not cloneable so each source has one explicit
/// owner and one unambiguous end-of-stream operation.
/// Call `close` or `blocking_close` explicitly; dropping the handle only makes
/// a best-effort non-blocking close request.
#[derive(Debug)]
pub struct AudioInput {
    source_id: AudioSourceId,
    commands: mpsc::Sender<SessionCommand>,
    closed: bool,
}

impl AudioInput {
    pub(crate) fn new(source_id: AudioSourceId, commands: mpsc::Sender<SessionCommand>) -> Self {
        Self {
            source_id,
            commands,
            closed: false,
        }
    }

    /// Identifier included in transcript segments produced by this input.
    pub const fn source_id(&self) -> AudioSourceId {
        self.source_id
    }

    /// Send one PCM chunk for this source.
    pub async fn send(&self, chunk: PcmChunk) -> Result<()> {
        self.commands
            .send(SessionCommand::Audio {
                source_id: self.source_id,
                chunk,
            })
            .await
            .map_err(|_| Error::StreamClosed)
    }

    /// Send one PCM chunk from a non-async capture thread.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_send(&self, chunk: PcmChunk) -> Result<()> {
        self.commands
            .blocking_send(SessionCommand::Audio {
                source_id: self.source_id,
                chunk,
            })
            .map_err(|_| Error::StreamClosed)
    }

    /// Finish and release this source after emitting any buffered segment.
    ///
    /// Transcript events must be drained concurrently because closing may
    /// emit a final segment through the bounded event channel.
    pub async fn close(mut self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(SessionCommand::CloseSource {
                source_id: self.source_id,
                reply: Some(reply_tx),
            })
            .await
            .map_err(|_| Error::StreamClosed)?;
        self.closed = true;
        reply_rx.await.map_err(|_| Error::StreamClosed)?
    }

    /// Finish and release this source from a non-async capture thread.
    ///
    /// Transcript events must be drained concurrently while this waits.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_close(mut self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .blocking_send(SessionCommand::CloseSource {
                source_id: self.source_id,
                reply: Some(reply_tx),
            })
            .map_err(|_| Error::StreamClosed)?;
        self.closed = true;
        reply_rx.blocking_recv().map_err(|_| Error::StreamClosed)?
    }
}

impl Drop for AudioInput {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.commands.try_send(SessionCommand::CloseSource {
                source_id: self.source_id,
                reply: None,
            });
        }
    }
}

/// Running Tokio session for one transcriber.
pub struct TranscriptionSession {
    /// Primary audio input, registered as `AudioSourceId::PRIMARY`.
    pub input: AudioInput,
    /// Receive transcript events or one terminal error here.
    ///
    /// After an error, the worker closes the event channel without emitting
    /// `TranscriptEvent::EndOfStream`.
    pub events: mpsc::Receiver<Result<TranscriptEvent>>,
    pub(crate) commands: mpsc::Sender<SessionCommand>,
    pub(crate) worker: TranscriptionWorker,
}

impl TranscriptionSession {
    /// Register and initialize an additional audio source.
    ///
    /// This waits for the source's VAD session to initialize. Closing a source
    /// releases its state and makes its capacity available to another source.
    pub async fn open_source(&self, config: AudioSourceConfig) -> Result<AudioInput> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(SessionCommand::OpenSource {
                config,
                reply: reply_tx,
            })
            .await
            .map_err(|_| Error::StreamClosed)?;
        let source_id = reply_rx.await.map_err(|_| Error::StreamClosed)??;
        Ok(AudioInput::new(source_id, self.commands.clone()))
    }

    /// Split the session into its primary input and output channel.
    ///
    /// Call `AudioInput::close` to flush the source and end the session.
    pub fn into_channels(self) -> (AudioInput, mpsc::Receiver<Result<TranscriptEvent>>) {
        (self.input, self.events)
    }

    /// Split the session into primary input, output, and worker handle.
    pub fn into_parts(
        self,
    ) -> (
        AudioInput,
        mpsc::Receiver<Result<TranscriptEvent>>,
        TranscriptionWorker,
    ) {
        (self.input, self.events, self.worker)
    }
}

pub(crate) enum SessionCommand {
    OpenSource {
        config: AudioSourceConfig,
        reply: oneshot::Sender<Result<AudioSourceId>>,
    },
    Audio {
        source_id: AudioSourceId,
        chunk: PcmChunk,
    },
    CloseSource {
        source_id: AudioSourceId,
        reply: Option<oneshot::Sender<Result<()>>>,
    },
}

pub(crate) fn run_transcription_worker(
    config: TranscriberConfig,
    mut model: Box<dyn AsrModel>,
    primary_stream: RecognitionStream,
    primary_vad: Box<dyn VadGate>,
    mut vad_factory: Box<dyn VadFactory>,
    mut command_rx: mpsc::Receiver<SessionCommand>,
    event_tx: mpsc::Sender<Result<TranscriptEvent>>,
) {
    let mut next_segment_id = 0u64;
    let mut next_source_id = 1u64;
    let mut sources = BTreeMap::from([(
        AudioSourceId::PRIMARY,
        SourceState {
            _config: AudioSourceConfig::other(),
            stream: primary_stream,
            vad: primary_vad,
            next_input_sample: 0,
        },
    )]);

    while let Some(command) = command_rx.blocking_recv() {
        match command {
            SessionCommand::OpenSource {
                config: source_config,
                reply,
            } => {
                if sources.len() >= config.max_sources {
                    let _ = reply.send(Err(Error::SourceLimit {
                        max_sources: config.max_sources,
                    }));
                    continue;
                }

                let vad = match vad_factory.create() {
                    Ok(vad) => vad,
                    Err(err) => {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                };
                let source_id = AudioSourceId::new(next_source_id);
                next_source_id = match next_source_id.checked_add(1) {
                    Some(value) => value,
                    None => {
                        let _ = reply.send(Err(Error::InvalidConfig(
                            "audio source identifier space exhausted".to_string(),
                        )));
                        continue;
                    }
                };
                if reply.send(Ok(source_id)).is_err() {
                    continue;
                }
                sources.insert(
                    source_id,
                    SourceState {
                        _config: source_config,
                        stream: RecognitionStream::default(),
                        vad,
                        next_input_sample: 0,
                    },
                );
            }
            SessionCommand::Audio { source_id, chunk } => {
                let Some(source) = sources.get_mut(&source_id) else {
                    fail(&event_tx, Error::SourceNotFound { source_id });
                    return;
                };
                if let Err(err) = process_chunk(
                    &config,
                    model.as_mut(),
                    source,
                    &event_tx,
                    &mut next_segment_id,
                    source_id,
                    chunk,
                ) {
                    fail(&event_tx, err);
                    return;
                }
            }
            SessionCommand::CloseSource { source_id, reply } => {
                let Some(source) = sources.remove(&source_id) else {
                    if let Some(reply) = reply {
                        let _ = reply.send(Err(Error::SourceNotFound { source_id }));
                    }
                    continue;
                };

                let result = finish_source(
                    &config,
                    model.as_mut(),
                    source,
                    &event_tx,
                    &mut next_segment_id,
                    source_id,
                );
                if let Some(reply) = reply {
                    let _ = reply.send(result.clone());
                }
                if let Err(err) = result {
                    fail(&event_tx, err);
                    return;
                }
                if sources.is_empty() {
                    let _ = send_event(&event_tx, Ok(TranscriptEvent::EndOfStream));
                    return;
                }
            }
        }
    }

    for (source_id, source) in sources {
        if let Err(err) = finish_source(
            &config,
            model.as_mut(),
            source,
            &event_tx,
            &mut next_segment_id,
            source_id,
        ) {
            fail(&event_tx, err);
            return;
        }
    }
    let _ = send_event(&event_tx, Ok(TranscriptEvent::EndOfStream));
}

struct SourceState {
    _config: AudioSourceConfig,
    stream: RecognitionStream,
    vad: Box<dyn VadGate>,
    next_input_sample: u64,
}

fn process_chunk(
    config: &TranscriberConfig,
    model: &mut dyn AsrModel,
    source: &mut SourceState,
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    next_segment_id: &mut u64,
    source_id: AudioSourceId,
    chunk: PcmChunk,
) -> Result<()> {
    chunk.format.validate()?;
    let start_sample = source.next_input_sample;
    source.next_input_sample = source
        .next_input_sample
        .saturating_add(chunk.samples.len() as u64);
    let speech_chunks = source.vad.push(&chunk, start_sample)?;
    handle_speech_chunks(
        config,
        model,
        &mut source.stream,
        event_tx,
        next_segment_id,
        source_id,
        speech_chunks,
    )
}

fn finish_source(
    config: &TranscriberConfig,
    model: &mut dyn AsrModel,
    mut source: SourceState,
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    next_segment_id: &mut u64,
    source_id: AudioSourceId,
) -> Result<()> {
    let final_chunks = source.vad.finish()?;
    handle_speech_chunks(
        config,
        model,
        &mut source.stream,
        event_tx,
        next_segment_id,
        source_id,
        final_chunks,
    )?;

    let started = Instant::now();
    let transcripts = source
        .stream
        .flush(model, &config.language, source.next_input_sample)?;
    send_transcripts(
        event_tx,
        next_segment_id,
        source_id,
        transcripts,
        started.elapsed(),
    )
}

fn handle_speech_chunks(
    config: &TranscriberConfig,
    model: &mut dyn AsrModel,
    stream: &mut RecognitionStream,
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    next_segment_id: &mut u64,
    source_id: AudioSourceId,
    chunks: Vec<SpeechChunk>,
) -> Result<()> {
    for speech in chunks {
        let started = Instant::now();
        let transcripts = stream.accept_speech(model, &speech, &config.language)?;
        send_transcripts(
            event_tx,
            next_segment_id,
            source_id,
            transcripts,
            started.elapsed(),
        )?;
    }
    Ok(())
}

fn send_transcripts(
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    next_segment_id: &mut u64,
    source_id: AudioSourceId,
    transcripts: Vec<BackendTranscript>,
    inference_duration: Duration,
) -> Result<()> {
    for transcript in transcripts {
        if transcript.text.trim().is_empty() {
            continue;
        }

        let id = SegmentId::new(*next_segment_id);
        *next_segment_id = next_segment_id.saturating_add(1);
        send_event(
            event_tx,
            Ok(TranscriptEvent::Segment(TranscriptSegment {
                id,
                source_id,
                speaker_id: None,
                text: transcript.text,
                start: duration_from_samples(transcript.start_sample),
                end: duration_from_samples(transcript.end_sample),
                inference_duration,
                is_final: transcript.is_final,
            })),
        )?;
    }
    Ok(())
}

fn send_event(
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    event: Result<TranscriptEvent>,
) -> Result<()> {
    event_tx
        .blocking_send(event)
        .map_err(|_| Error::StreamClosed)
}

fn fail(event_tx: &mpsc::Sender<Result<TranscriptEvent>>, err: Error) {
    let _ = send_event(event_tx, Err(err));
}
