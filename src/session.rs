use crate::PCM_SAMPLE_RATE_HZ;
use crate::backend::AsrModel;
use crate::event::{SegmentId, TranscriptEvent, TranscriptSegment};
use crate::vad::{SpeechChunk, VadFactory, VadGate, duration_from_samples};
use crate::{AudioSourceId, Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Join handle for the blocking transcription worker.
pub type TranscriptionWorker = JoinHandle<()>;

/// A point-in-time snapshot of transcript output delivery statistics.
///
/// All transcript segments, terminal errors, and end-of-stream markers count
/// as events. Counters saturate instead of wrapping when their numeric range is
/// exhausted.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputMetrics {
    /// Events currently waiting to be received.
    pub pending_events: usize,
    /// Largest observed number of simultaneously pending events.
    pub peak_pending_events: usize,
    /// Events successfully added to the output queue.
    pub emitted_events: u64,
    /// Events returned through the receiver API.
    pub received_events: u64,
    /// Queued events abandoned when the receiver was dropped.
    pub discarded_events: u64,
    /// Event delivery attempts rejected after the receiver closed.
    pub delivery_failures: u64,
    /// Whether the consumer explicitly closed or dropped the receiver.
    pub receiver_closed: bool,
}

#[derive(Debug)]
struct OutputState {
    counters: Mutex<OutputMetrics>,
}

impl Default for OutputState {
    fn default() -> Self {
        Self {
            counters: Mutex::new(OutputMetrics {
                pending_events: 0,
                peak_pending_events: 0,
                emitted_events: 0,
                received_events: 0,
                discarded_events: 0,
                delivery_failures: 0,
                receiver_closed: false,
            }),
        }
    }
}

impl OutputState {
    fn lock(&self) -> MutexGuard<'_, OutputMetrics> {
        self.counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn metrics(&self) -> OutputMetrics {
        *self.lock()
    }

    fn record_received(&self, count: usize) {
        if count == 0 {
            return;
        }
        let mut counters = self.lock();
        counters.pending_events = counters.pending_events.saturating_sub(count);
        counters.received_events = counters.received_events.saturating_add(count as u64);
    }

    fn record_discarded(&self, count: usize) {
        let mut counters = self.lock();
        counters.receiver_closed = true;
        counters.pending_events = counters.pending_events.saturating_sub(count);
        counters.discarded_events = counters.discarded_events.saturating_add(count as u64);
    }

    fn record_receiver_closed(&self) {
        self.lock().receiver_closed = true;
    }
}

/// Cloneable observer for transcript output delivery statistics.
///
/// A monitor only retains the metrics state. It does not keep the event queue,
/// audio inputs, or transcription worker alive.
#[derive(Clone, Debug)]
pub struct OutputMonitor {
    state: Arc<OutputState>,
}

impl OutputMonitor {
    /// Return the latest output delivery statistics.
    pub fn metrics(&self) -> OutputMetrics {
        self.state.metrics()
    }
}

/// Receiver for transcript events produced by a transcription session.
///
/// Output remains unbounded so closing audio inputs never depends on draining
/// this receiver concurrently. Applications should use [`Self::monitor`] to
/// detect a growing backlog and continuously drain long-running sessions.
pub struct TranscriptEventReceiver {
    inner: mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
    monitor: OutputMonitor,
    cancelled: Arc<AtomicBool>,
    commands: mpsc::WeakSender<SessionCommand>,
}

impl TranscriptEventReceiver {
    fn new(
        inner: mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
        state: Arc<OutputState>,
        cancelled: Arc<AtomicBool>,
        commands: mpsc::WeakSender<SessionCommand>,
    ) -> Self {
        Self {
            inner,
            monitor: OutputMonitor { state },
            cancelled,
            commands,
        }
    }

    /// Receive the next transcript event, waiting if necessary.
    pub async fn recv(&mut self) -> Option<Result<TranscriptEvent>> {
        let event = self.inner.recv().await;
        self.record_one(&event);
        event
    }

    /// Receive up to `limit` transcript events into `buffer`.
    pub async fn recv_many(
        &mut self,
        buffer: &mut Vec<Result<TranscriptEvent>>,
        limit: usize,
    ) -> usize {
        let count = self.inner.recv_many(buffer, limit).await;
        self.monitor.state.record_received(count);
        count
    }

    /// Try to receive the next transcript event without waiting.
    pub fn try_recv(
        &mut self,
    ) -> std::result::Result<Result<TranscriptEvent>, mpsc::error::TryRecvError> {
        let result = self.inner.try_recv();
        if result.is_ok() {
            self.monitor.state.record_received(1);
        }
        result
    }

    /// Receive the next transcript event from a synchronous context.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_recv(&mut self) -> Option<Result<TranscriptEvent>> {
        let event = self.inner.blocking_recv();
        self.record_one(&event);
        event
    }

    /// Receive up to `limit` transcript events from a synchronous context.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_recv_many(
        &mut self,
        buffer: &mut Vec<Result<TranscriptEvent>>,
        limit: usize,
    ) -> usize {
        let count = self.inner.blocking_recv_many(buffer, limit);
        self.monitor.state.record_received(count);
        count
    }

    /// Poll for the next transcript event.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<TranscriptEvent>>> {
        let event = self.inner.poll_recv(cx);
        if matches!(&event, Poll::Ready(Some(_))) {
            self.monitor.state.record_received(1);
        }
        event
    }

    /// Poll for up to `limit` transcript events, extending `buffer`.
    pub fn poll_recv_many(
        &mut self,
        cx: &mut Context<'_>,
        buffer: &mut Vec<Result<TranscriptEvent>>,
        limit: usize,
    ) -> Poll<usize> {
        let result = self.inner.poll_recv_many(cx, buffer, limit);
        if let Poll::Ready(count) = result {
            self.monitor.state.record_received(count);
        }
        result
    }

    /// Stop new output and cancel the worker while retaining queued events.
    ///
    /// Call [`Self::recv`] until it returns `None` to drain events that were
    /// already queued when the receiver closed.
    pub fn close(&mut self) {
        self.inner.close();
        self.cancel_worker();
        self.monitor.state.record_receiver_closed();
    }

    /// Return the number of events currently waiting to be received.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return whether no events are currently waiting to be received.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return whether the output channel is closed to new events.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Return the latest output delivery statistics.
    pub fn metrics(&self) -> OutputMetrics {
        self.monitor.metrics()
    }

    /// Create an observer that can inspect statistics from another task.
    pub fn monitor(&self) -> OutputMonitor {
        self.monitor.clone()
    }

    fn record_one(&self, event: &Option<Result<TranscriptEvent>>) {
        if event.is_some() {
            self.monitor.state.record_received(1);
        }
    }

    fn cancel_worker(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel)
            && let Some(commands) = self.commands.upgrade()
        {
            let _ = commands.try_send(SessionCommand::Cancel);
        }
    }
}

impl Drop for TranscriptEventReceiver {
    fn drop(&mut self) {
        self.inner.close();
        self.cancel_worker();
        let discarded = self.inner.len();
        self.monitor.state.record_discarded(discarded);
    }
}

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
    cancelled: Arc<AtomicBool>,
    closed: bool,
}

impl AudioInput {
    pub(crate) fn new(
        source_id: AudioSourceId,
        commands: mpsc::Sender<SessionCommand>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            source_id,
            commands,
            cancelled,
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
        self.ensure_active()?;
        self.commands
            .send(SessionCommand::Audio {
                source_id: self.source_id,
                timestamp,
                samples,
            })
            .await
            .map_err(|_| Error::StreamClosed)?;
        self.ensure_active()
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
        self.ensure_active()?;
        self.commands
            .blocking_send(SessionCommand::Audio {
                source_id: self.source_id,
                timestamp,
                samples,
            })
            .map_err(|_| Error::StreamClosed)?;
        self.ensure_active()
    }

    /// Finish and release this source after emitting any buffered segment.
    ///
    pub async fn close(mut self) -> Result<()> {
        self.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(SessionCommand::CloseSource {
                source_id: self.source_id,
                reply: Some(reply_tx),
            })
            .await
            .map_err(|_| Error::StreamClosed)?;
        self.closed = true;
        let result = reply_rx.await.map_err(|_| Error::StreamClosed)?;
        self.ensure_active()?;
        result
    }

    /// Finish and release this source from a non-async capture thread.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_close(mut self) -> Result<()> {
        self.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .blocking_send(SessionCommand::CloseSource {
                source_id: self.source_id,
                reply: Some(reply_tx),
            })
            .map_err(|_| Error::StreamClosed)?;
        self.closed = true;
        let result = reply_rx.blocking_recv().map_err(|_| Error::StreamClosed)?;
        self.ensure_active()?;
        result
    }

    fn ensure_active(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(Error::StreamClosed)
        } else {
            Ok(())
        }
    }
}

impl Drop for AudioInput {
    fn drop(&mut self) {
        if !self.closed && !self.cancelled.load(Ordering::Acquire) {
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
    pub events: TranscriptEventReceiver,
    pub(crate) worker: TranscriptionWorker,
}

impl TranscriptionSession {
    /// Register and initialize an additional audio source.
    ///
    /// This waits for the source's VAD session to initialize. Closing a source
    /// releases its state and makes its capacity available to another source.
    pub async fn open_source(&self) -> Result<AudioInput> {
        self.input.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.input
            .commands
            .send(SessionCommand::OpenSource { reply: reply_tx })
            .await
            .map_err(|_| Error::StreamClosed)?;
        let result = reply_rx.await.map_err(|_| Error::StreamClosed)?;
        self.input.ensure_active()?;
        let source_id = result?;
        Ok(AudioInput::new(
            source_id,
            self.input.commands.clone(),
            Arc::clone(&self.input.cancelled),
        ))
    }

    /// Split the session into primary input, output, and worker handle.
    pub fn into_parts(self) -> (AudioInput, TranscriptEventReceiver, TranscriptionWorker) {
        (self.input, self.events, self.worker)
    }
}

pub(crate) enum SessionCommand {
    Cancel,
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

pub(crate) struct EventSender {
    inner: mpsc::UnboundedSender<Result<TranscriptEvent>>,
    state: Arc<OutputState>,
}

impl EventSender {
    fn send(&self, event: Result<TranscriptEvent>) -> Result<()> {
        // Hold the accounting lock across the non-blocking send so a receiver
        // cannot permanently decrement pending before this send increments it.
        let mut counters = self.state.lock();
        match self.inner.send(event) {
            Ok(()) => {
                counters.pending_events = counters.pending_events.saturating_add(1);
                counters.peak_pending_events =
                    counters.peak_pending_events.max(counters.pending_events);
                counters.emitted_events = counters.emitted_events.saturating_add(1);
                Ok(())
            }
            Err(_) => {
                counters.delivery_failures = counters.delivery_failures.saturating_add(1);
                Err(Error::StreamClosed)
            }
        }
    }
}

pub(crate) fn output_channel(
    commands: mpsc::WeakSender<SessionCommand>,
) -> (EventSender, TranscriptEventReceiver, Arc<AtomicBool>) {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let state = Arc::new(OutputState::default());
    let cancelled = Arc::new(AtomicBool::new(false));
    (
        EventSender {
            inner: event_tx,
            state: Arc::clone(&state),
        },
        TranscriptEventReceiver::new(event_rx, state, Arc::clone(&cancelled), commands),
        cancelled,
    )
}

pub(crate) fn run_transcription_worker(
    max_sources: usize,
    mut model: Box<dyn AsrModel>,
    primary_vad: Box<dyn VadGate>,
    mut vad_factory: Box<dyn VadFactory>,
    mut command_rx: mpsc::Receiver<SessionCommand>,
    event_tx: EventSender,
    cancelled: Arc<AtomicBool>,
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

    loop {
        if cancelled.load(Ordering::Acquire) {
            return;
        }
        let Some(command) = command_rx.blocking_recv() else {
            if cancelled.load(Ordering::Acquire) {
                return;
            }
            break;
        };
        if cancelled.load(Ordering::Acquire) {
            return;
        }
        match command {
            SessionCommand::Cancel => return,
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
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
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
                    cancelled: cancelled.as_ref(),
                };
                let result = process_chunk(
                    model.as_mut(),
                    source,
                    &mut sink,
                    source_id,
                    timestamp,
                    samples,
                );
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
                if let Err(err) = result {
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
                    cancelled: cancelled.as_ref(),
                };
                let result = finish_source(model.as_mut(), source, &mut sink, source_id);
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
                if let Some(reply) = reply {
                    let _ = reply.send(result.clone());
                }
                if let Err(err) = result {
                    fail(&event_tx, err);
                    return;
                }
                if sources.is_empty() {
                    let _ = event_tx.send(Ok(TranscriptEvent::EndOfStream));
                    return;
                }
            }
        }
    }

    if cancelled.load(Ordering::Acquire) {
        return;
    }
    for (source_id, source) in sources {
        if cancelled.load(Ordering::Acquire) {
            return;
        }
        let mut sink = EventSink {
            event_tx: &event_tx,
            next_segment_id: &mut next_segment_id,
            cancelled: cancelled.as_ref(),
        };
        if let Err(err) = finish_source(model.as_mut(), source, &mut sink, source_id) {
            fail(&event_tx, err);
            return;
        }
    }
    let _ = event_tx.send(Ok(TranscriptEvent::EndOfStream));
}

struct SourceState {
    vad: Box<dyn VadGate>,
    next_input_sample: u64,
    timeline_offset_sample: Option<u64>,
}

struct EventSink<'a> {
    event_tx: &'a EventSender,
    next_segment_id: &'a mut u64,
    cancelled: &'a AtomicBool,
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
    if sink.cancelled.load(Ordering::Acquire) {
        return Ok(());
    }
    let final_chunks = source.vad.finish()?;
    if sink.cancelled.load(Ordering::Acquire) {
        return Ok(());
    }
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
        if sink.cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        if speech.samples.is_empty() {
            continue;
        }
        let started = Instant::now();
        let text = model.transcribe(speech.samples)?;
        if sink.cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
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
    sink.event_tx
        .send(Ok(TranscriptEvent::Segment(TranscriptSegment {
            id,
            source_id,
            speaker_id: None,
            text,
            start: duration_from_samples(start_sample),
            end: duration_from_samples(end_sample),
            inference_duration,
            is_final: true,
        })))
}

fn fail(event_tx: &EventSender, err: Error) {
    let _ = event_tx.send(Err(err));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::poll_fn;

    fn test_output_channel() -> (
        EventSender,
        TranscriptEventReceiver,
        OutputMonitor,
        mpsc::Sender<SessionCommand>,
    ) {
        let (command_tx, _command_rx) = mpsc::channel(4);
        let (event_tx, events, _cancelled) = output_channel(command_tx.downgrade());
        let monitor = events.monitor();
        (event_tx, events, monitor, command_tx)
    }

    fn end_event() -> Result<TranscriptEvent> {
        Ok(TranscriptEvent::EndOfStream)
    }

    #[tokio::test]
    async fn receiver_metrics_track_async_receive_paths() {
        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        for _ in 0..6 {
            event_tx.send(end_event()).unwrap();
        }
        event_tx
            .send(Err(Error::Backend("test failure".to_string())))
            .unwrap();

        assert_eq!(
            monitor.metrics(),
            OutputMetrics {
                pending_events: 7,
                peak_pending_events: 7,
                emitted_events: 7,
                received_events: 0,
                discarded_events: 0,
                delivery_failures: 0,
                receiver_closed: false,
            }
        );

        assert!(events.recv().await.unwrap().is_ok());
        assert!(events.try_recv().unwrap().is_ok());

        let mut received = Vec::new();
        assert_eq!(events.recv_many(&mut received, 2).await, 2);
        assert_eq!(received.len(), 2);

        assert!(poll_fn(|cx| events.poll_recv(cx)).await.unwrap().is_ok());
        assert_eq!(
            poll_fn(|cx| events.poll_recv_many(cx, &mut received, 2)).await,
            2
        );
        assert_eq!(received.len(), 4);
        assert!(matches!(
            received.last(),
            Some(Err(Error::Backend(message))) if message == "test failure"
        ));

        let metrics = events.metrics();
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.peak_pending_events, 7);
        assert_eq!(metrics.emitted_events, 7);
        assert_eq!(metrics.received_events, 7);
        assert_eq!(metrics.discarded_events, 0);
        assert_eq!(metrics.delivery_failures, 0);
        assert!(!metrics.receiver_closed);
    }

    #[test]
    fn receiver_metrics_track_blocking_receive_paths() {
        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        for _ in 0..3 {
            event_tx.send(end_event()).unwrap();
        }

        assert!(events.blocking_recv().unwrap().is_ok());
        let mut received = Vec::new();
        assert_eq!(events.blocking_recv_many(&mut received, 2), 2);

        let metrics = monitor.metrics();
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.received_events, 3);
        assert_eq!(metrics.peak_pending_events, 3);
    }

    #[tokio::test]
    async fn closing_receiver_cancels_output_but_keeps_queued_events() {
        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        event_tx.send(end_event()).unwrap();
        event_tx.send(end_event()).unwrap();

        events.close();
        assert!(event_tx.send(end_event()).is_err());

        let mut received = Vec::new();
        assert_eq!(events.recv_many(&mut received, 8).await, 2);
        assert_eq!(events.recv().await, None);

        let metrics = monitor.metrics();
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.emitted_events, 2);
        assert_eq!(metrics.received_events, 2);
        assert_eq!(metrics.discarded_events, 0);
        assert_eq!(metrics.delivery_failures, 1);
        assert!(metrics.receiver_closed);
    }

    #[test]
    fn dropping_receiver_counts_discarded_events() {
        let (event_tx, events, monitor, _command_tx) = test_output_channel();
        event_tx.send(end_event()).unwrap();
        event_tx.send(end_event()).unwrap();

        drop(events);
        assert!(event_tx.send(end_event()).is_err());

        let metrics = monitor.metrics();
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.peak_pending_events, 2);
        assert_eq!(metrics.emitted_events, 2);
        assert_eq!(metrics.received_events, 0);
        assert_eq!(metrics.discarded_events, 2);
        assert_eq!(metrics.delivery_failures, 1);
        assert!(metrics.receiver_closed);
    }

    #[test]
    fn closing_receiver_sets_cancellation_before_metrics_lock() {
        let (command_tx, _command_rx) = mpsc::channel(1);
        let (_event_tx, mut events, cancelled) = output_channel(command_tx.downgrade());
        let monitor = events.monitor();
        let metrics_guard = monitor.state.lock();
        let (started_tx, started_rx) = std::sync::mpsc::channel();

        let closer = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            events.close();
            events
        });
        started_rx.recv().unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while !cancelled.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let cancelled_before_unlock = cancelled.load(Ordering::Acquire);
        drop(metrics_guard);
        drop(closer.join().unwrap());

        assert!(cancelled_before_unlock);
    }

    #[tokio::test]
    async fn input_close_observes_cancellation_after_worker_reply() {
        let (command_tx, mut command_rx) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let input = AudioInput::new(AudioSourceId::PRIMARY, command_tx, Arc::clone(&cancelled));

        let responder = tokio::spawn(async move {
            let SessionCommand::CloseSource { reply, .. } = command_rx.recv().await.unwrap() else {
                panic!("expected close command");
            };
            cancelled.store(true, Ordering::Release);
            reply.unwrap().send(Ok(())).unwrap();
        });

        assert_eq!(input.close().await, Err(Error::StreamClosed));
        responder.await.unwrap();
    }

    #[test]
    fn output_metrics_counters_saturate_instead_of_wrapping() {
        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        {
            let mut counters = monitor.state.lock();
            counters.pending_events = usize::MAX;
            counters.peak_pending_events = usize::MAX;
            counters.emitted_events = u64::MAX;
            counters.received_events = u64::MAX;
            counters.discarded_events = u64::MAX;
            counters.delivery_failures = u64::MAX;
        }

        event_tx.send(end_event()).unwrap();
        assert!(events.try_recv().unwrap().is_ok());
        events.close();
        assert!(event_tx.send(end_event()).is_err());
        drop(events);

        let metrics = monitor.metrics();
        assert_eq!(metrics.pending_events, usize::MAX - 1);
        assert_eq!(metrics.peak_pending_events, usize::MAX);
        assert_eq!(metrics.emitted_events, u64::MAX);
        assert_eq!(metrics.received_events, u64::MAX);
        assert_eq!(metrics.discarded_events, u64::MAX);
        assert_eq!(metrics.delivery_failures, u64::MAX);
        assert!(metrics.receiver_closed);
    }

    #[test]
    fn concurrent_send_and_receive_keep_metrics_consistent() {
        const EVENT_COUNT: usize = 10_000;

        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        let sender = std::thread::spawn(move || {
            for _ in 0..EVENT_COUNT {
                event_tx.send(end_event()).unwrap();
            }
        });

        for _ in 0..EVENT_COUNT {
            assert!(events.blocking_recv().unwrap().is_ok());
        }
        sender.join().unwrap();

        let metrics = monitor.metrics();
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.emitted_events, EVENT_COUNT as u64);
        assert_eq!(metrics.received_events, EVENT_COUNT as u64);
        assert!(metrics.peak_pending_events >= 1);
        assert_eq!(metrics.delivery_failures, 0);
    }

    #[test]
    fn receiver_and_monitor_do_not_add_strong_command_senders() {
        let (command_tx, _command_rx) = mpsc::channel(1);
        assert_eq!(command_tx.strong_count(), 1);

        let (_event_tx, events, _cancelled) = output_channel(command_tx.downgrade());
        let monitor = events.monitor();
        assert_eq!(command_tx.strong_count(), 1);

        drop(events);
        drop(monitor);
        assert_eq!(command_tx.strong_count(), 1);
    }
}
