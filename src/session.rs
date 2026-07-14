use crate::PCM_SAMPLE_RATE_HZ;
use crate::backend::AsrModel;
use crate::event::{SegmentId, TranscriptEvent, TranscriptSegment};
use crate::vad::{SpeechChunk, VadFactory, VadGate, duration_from_samples};
use crate::{AudioSourceId, Error, Result};
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

    /// Send one f32 mono 16 kHz PCM chunk for this source.
    ///
    /// The first un-timestamped chunk anchors the source at session time zero;
    /// later chunks continue from the preceding sample count.
    pub async fn send(&self, samples: Vec<f32>) -> Result<()> {
        self.send_command(None, samples).await
    }

    /// Send an f32 mono 16 kHz chunk anchored to the session timeline.
    ///
    /// The timestamp is rounded down to the nearest 16 kHz sample boundary.
    /// The first explicit timestamp anchors this source; later explicit
    /// timestamps must equal the position implied by all previously sent
    /// samples after the same quantization.
    /// Timestamp validation failures are emitted as terminal errors through
    /// `TranscriptionSession::events` after this command is accepted.
    pub async fn send_at(&self, timestamp: Duration, samples: Vec<f32>) -> Result<()> {
        self.send_command(Some(timestamp), samples).await
    }

    async fn send_command(&self, timestamp: Option<Duration>, samples: Vec<f32>) -> Result<()> {
        self.commands
            .send(SessionCommand::Audio {
                source_id: self.source_id,
                timestamp,
                samples,
            })
            .await
            .map_err(|_| Error::StreamClosed)
    }

    /// Send one f32 mono 16 kHz PCM chunk from a non-async capture thread.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_send(&self, samples: Vec<f32>) -> Result<()> {
        self.blocking_send_command(None, samples)
    }

    /// Send a timestamped chunk from a non-async capture thread.
    ///
    /// This has the same timeline requirements and terminal error reporting as
    /// `send_at`.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_send_at(&self, timestamp: Duration, samples: Vec<f32>) -> Result<()> {
        self.blocking_send_command(Some(timestamp), samples)
    }

    fn blocking_send_command(&self, timestamp: Option<Duration>, samples: Vec<f32>) -> Result<()> {
        self.commands
            .blocking_send(SessionCommand::Audio {
                source_id: self.source_id,
                timestamp,
                samples,
            })
            .map_err(|_| Error::StreamClosed)
    }

    /// Finish and release this source after emitting any buffered segment.
    ///
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
    pub events: mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
    pub(crate) commands: mpsc::Sender<SessionCommand>,
    pub(crate) worker: TranscriptionWorker,
}

impl TranscriptionSession {
    /// Register and initialize an additional audio source.
    ///
    /// This waits for the source's VAD session to initialize. Closing a source
    /// releases its state and makes its capacity available to another source.
    pub async fn open_source(&self) -> Result<AudioInput> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(SessionCommand::OpenSource { reply: reply_tx })
            .await
            .map_err(|_| Error::StreamClosed)?;
        let source_id = reply_rx.await.map_err(|_| Error::StreamClosed)??;
        Ok(AudioInput::new(source_id, self.commands.clone()))
    }

    /// Split the session into primary input, output, and worker handle.
    pub fn into_parts(
        self,
    ) -> (
        AudioInput,
        mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
        TranscriptionWorker,
    ) {
        (self.input, self.events, self.worker)
    }
}

pub(crate) enum SessionCommand {
    OpenSource {
        reply: oneshot::Sender<Result<AudioSourceId>>,
    },
    Audio {
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
    },
    CloseSource {
        source_id: AudioSourceId,
        reply: Option<oneshot::Sender<Result<()>>>,
    },
}

pub(crate) fn run_transcription_worker(
    max_sources: usize,
    mut model: Box<dyn AsrModel>,
    primary_vad: Box<dyn VadGate>,
    mut vad_factory: Box<dyn VadFactory>,
    mut command_rx: mpsc::Receiver<SessionCommand>,
    event_tx: mpsc::UnboundedSender<Result<TranscriptEvent>>,
) {
    let mut next_segment_id = 0u64;
    let mut next_source_id = 1u64;
    let mut sources = vec![(
        AudioSourceId::PRIMARY,
        SourceState {
            vad: primary_vad,
            next_input_sample: 0,
            timeline_offset_sample: None,
        },
    )];

    while let Some(command) = command_rx.blocking_recv() {
        match command {
            SessionCommand::OpenSource { reply } => {
                if sources.len() >= max_sources {
                    let _ = reply.send(Err(Error::SourceLimit { max_sources }));
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
                sources.push((
                    source_id,
                    SourceState {
                        vad,
                        next_input_sample: 0,
                        timeline_offset_sample: None,
                    },
                ));
            }
            SessionCommand::Audio {
                source_id,
                timestamp,
                samples,
            } => {
                let Some(source) = sources
                    .iter_mut()
                    .find(|(id, _)| *id == source_id)
                    .map(|(_, source)| source)
                else {
                    fail(&event_tx, Error::SourceNotFound { source_id });
                    return;
                };
                let mut sink = EventSink {
                    event_tx: &event_tx,
                    next_segment_id: &mut next_segment_id,
                };
                if let Err(err) = process_chunk(
                    model.as_mut(),
                    source,
                    &mut sink,
                    source_id,
                    timestamp,
                    samples,
                ) {
                    fail(&event_tx, err);
                    return;
                }
            }
            SessionCommand::CloseSource { source_id, reply } => {
                let Some(position) = sources.iter().position(|(id, _)| *id == source_id) else {
                    if let Some(reply) = reply {
                        let _ = reply.send(Err(Error::SourceNotFound { source_id }));
                    }
                    continue;
                };
                let (_, source) = sources.remove(position);

                let mut sink = EventSink {
                    event_tx: &event_tx,
                    next_segment_id: &mut next_segment_id,
                };
                let result = finish_source(model.as_mut(), source, &mut sink, source_id);
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
        let mut sink = EventSink {
            event_tx: &event_tx,
            next_segment_id: &mut next_segment_id,
        };
        if let Err(err) = finish_source(model.as_mut(), source, &mut sink, source_id) {
            fail(&event_tx, err);
            return;
        }
    }
    let _ = send_event(&event_tx, Ok(TranscriptEvent::EndOfStream));
}

struct SourceState {
    vad: Box<dyn VadGate>,
    next_input_sample: u64,
    timeline_offset_sample: Option<u64>,
}

struct EventSink<'a> {
    event_tx: &'a mpsc::UnboundedSender<Result<TranscriptEvent>>,
    next_segment_id: &'a mut u64,
}

fn resolve_timeline_offset(
    source_id: AudioSourceId,
    source: &mut SourceState,
    timestamp: Option<Duration>,
) -> Result<u64> {
    let explicit_sample = timestamp
        .map(|timestamp| session_sample_from_duration(source_id, timestamp))
        .transpose()?;

    match source.timeline_offset_sample {
        Some(offset) => {
            if let Some(actual_sample) = explicit_sample {
                let expected_sample = timeline_sample(source_id, offset, source.next_input_sample)?;
                if actual_sample != expected_sample {
                    return Err(Error::TimestampDiscontinuity {
                        source_id,
                        expected: duration_from_samples(expected_sample),
                        actual: timestamp.expect("explicit sample requires a timestamp"),
                    });
                }
            }
            Ok(offset)
        }
        None => {
            let offset = explicit_sample.unwrap_or(0);
            source.timeline_offset_sample = Some(offset);
            Ok(offset)
        }
    }
}

fn session_sample_from_duration(source_id: AudioSourceId, timestamp: Duration) -> Result<u64> {
    let scaled = timestamp.as_nanos() * PCM_SAMPLE_RATE_HZ as u128;
    u64::try_from(scaled / 1_000_000_000).map_err(|_| Error::InvalidTimestamp {
        source_id,
        timestamp,
        message: "timestamp exceeds the supported session timeline".to_string(),
    })
}

fn timeline_sample(source_id: AudioSourceId, offset: u64, sample: u64) -> Result<u64> {
    offset
        .checked_add(sample)
        .ok_or_else(|| Error::InvalidTimestamp {
            source_id,
            timestamp: duration_from_samples(offset),
            message: "timestamp plus source audio length overflows the session timeline"
                .to_string(),
        })
}

fn process_chunk(
    model: &mut dyn AsrModel,
    source: &mut SourceState,
    sink: &mut EventSink<'_>,
    source_id: AudioSourceId,
    timestamp: Option<Duration>,
    samples: Vec<f32>,
) -> Result<()> {
    let timeline_offset_sample = resolve_timeline_offset(source_id, source, timestamp)?;
    let start_sample = source.next_input_sample;
    source.next_input_sample = source
        .next_input_sample
        .saturating_add(samples.len() as u64);
    let speech_chunks = source.vad.push(&samples, start_sample)?;
    handle_speech_chunks(
        model,
        sink,
        source_id,
        timeline_offset_sample,
        speech_chunks,
    )
}

fn finish_source(
    model: &mut dyn AsrModel,
    mut source: SourceState,
    sink: &mut EventSink<'_>,
    source_id: AudioSourceId,
) -> Result<()> {
    let final_chunks = source.vad.finish()?;
    let timeline_offset_sample = source.timeline_offset_sample.unwrap_or(0);
    handle_speech_chunks(model, sink, source_id, timeline_offset_sample, final_chunks)
}

fn handle_speech_chunks(
    model: &mut dyn AsrModel,
    sink: &mut EventSink<'_>,
    source_id: AudioSourceId,
    timeline_offset_sample: u64,
    chunks: Vec<SpeechChunk>,
) -> Result<()> {
    for speech in chunks {
        if speech.samples.is_empty() {
            continue;
        }
        let started = Instant::now();
        let text = model.transcribe(speech.samples)?;
        if text.trim().is_empty() {
            continue;
        }
        send_transcript(
            sink,
            source_id,
            timeline_offset_sample,
            speech.start_sample,
            speech.end_sample,
            text,
            started.elapsed(),
        )?;
    }
    Ok(())
}

fn send_transcript(
    sink: &mut EventSink<'_>,
    source_id: AudioSourceId,
    timeline_offset_sample: u64,
    transcript_start_sample: u64,
    transcript_end_sample: u64,
    text: String,
    inference_duration: Duration,
) -> Result<()> {
    let id = SegmentId::new(*sink.next_segment_id);
    *sink.next_segment_id = sink.next_segment_id.saturating_add(1);
    let start_sample = timeline_sample(source_id, timeline_offset_sample, transcript_start_sample)?;
    let end_sample = timeline_sample(source_id, timeline_offset_sample, transcript_end_sample)?;
    send_event(
        sink.event_tx,
        Ok(TranscriptEvent::Segment(TranscriptSegment {
            id,
            source_id,
            speaker_id: None,
            text,
            start: duration_from_samples(start_sample),
            end: duration_from_samples(end_sample),
            inference_duration,
            is_final: true,
        })),
    )
}

fn send_event(
    event_tx: &mpsc::UnboundedSender<Result<TranscriptEvent>>,
    event: Result<TranscriptEvent>,
) -> Result<()> {
    event_tx.send(event).map_err(|_| Error::StreamClosed)
}

fn fail(event_tx: &mpsc::UnboundedSender<Result<TranscriptEvent>>, err: Error) {
    let _ = send_event(event_tx, Err(err));
}
