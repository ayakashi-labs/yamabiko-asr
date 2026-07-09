use crate::backend::{BackendTranscript, StreamingAsrBackend};
use crate::event::{TranscriptEvent, TranscriptSegment};
use crate::vad::{SpeechChunk, VadGate, duration_from_samples};
use crate::{Error, PcmChunk, Result, TranscriberConfig};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Running Tokio session for one transcriber.
pub struct TranscriptionSession {
    /// Send f32 mono 16 kHz PCM chunks here.
    pub input: mpsc::Sender<PcmChunk>,
    /// Receive transcript events and recoverable/fatal errors here.
    pub events: mpsc::Receiver<Result<TranscriptEvent>>,
    pub(crate) worker: JoinHandle<()>,
}

impl TranscriptionSession {
    /// Split the session into its input and output channels.
    ///
    /// Dropping the returned input sender closes the stream. The worker is
    /// detached; it exits after emitting `TranscriptEvent::EndOfStream` or an
    /// error when all senders are dropped.
    pub fn into_channels(
        self,
    ) -> (
        mpsc::Sender<PcmChunk>,
        mpsc::Receiver<Result<TranscriptEvent>>,
    ) {
        (self.input, self.events)
    }

    /// Wait for the blocking transcription worker to exit.
    pub async fn join(self) -> Result<()> {
        self.worker
            .await
            .map_err(|err| Error::Join(err.to_string()))
    }
}

pub(crate) fn run_transcription_worker(
    config: TranscriberConfig,
    mut backend: Box<dyn StreamingAsrBackend>,
    mut vad: Box<dyn VadGate>,
    mut input_rx: mpsc::Receiver<PcmChunk>,
    event_tx: mpsc::Sender<Result<TranscriptEvent>>,
) {
    let mut next_input_sample = 0u64;

    while let Some(chunk) = input_rx.blocking_recv() {
        if let Err(err) = chunk.format.validate() {
            send(&event_tx, Err(err));
            return;
        }

        let start_sample = next_input_sample;
        next_input_sample = next_input_sample.saturating_add(chunk.samples.len() as u64);

        let speech_chunks = match vad.push(&chunk, start_sample) {
            Ok(chunks) => chunks,
            Err(err) => {
                send(&event_tx, Err(err));
                return;
            }
        };

        if !handle_speech_chunks(&config, backend.as_mut(), &event_tx, speech_chunks) {
            return;
        }
    }

    let final_chunks = match vad.finish() {
        Ok(chunks) => chunks,
        Err(err) => {
            send(&event_tx, Err(err));
            return;
        }
    };

    if !handle_speech_chunks(&config, backend.as_mut(), &event_tx, final_chunks) {
        return;
    }

    match backend.flush(next_input_sample) {
        Ok(transcripts) => {
            if !send_transcripts(&event_tx, transcripts) {
                return;
            }
        }
        Err(err) => {
            send(&event_tx, Err(err));
            return;
        }
    }

    send(&event_tx, Ok(TranscriptEvent::EndOfStream));
}

fn handle_speech_chunks(
    config: &TranscriberConfig,
    backend: &mut dyn StreamingAsrBackend,
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    chunks: Vec<SpeechChunk>,
) -> bool {
    for speech in chunks {
        let transcripts = match backend.accept_speech(&speech, &config.language) {
            Ok(transcripts) => transcripts,
            Err(err) => {
                send(event_tx, Err(err));
                return false;
            }
        };

        if !send_transcripts(event_tx, transcripts) {
            return false;
        }
    }

    true
}

fn send_transcripts(
    event_tx: &mpsc::Sender<Result<TranscriptEvent>>,
    transcripts: Vec<BackendTranscript>,
) -> bool {
    for transcript in transcripts {
        if transcript.text.trim().is_empty() {
            continue;
        }

        let segment = TranscriptSegment {
            text: transcript.text,
            start: duration_from_samples(transcript.start_sample),
            end: duration_from_samples(transcript.end_sample),
            is_final: transcript.is_final,
        };

        if !send(event_tx, Ok(TranscriptEvent::Segment(segment))) {
            return false;
        }
    }

    true
}

fn send(event_tx: &mpsc::Sender<Result<TranscriptEvent>>, event: Result<TranscriptEvent>) -> bool {
    event_tx.blocking_send(event).is_ok()
}
