use crate::PCM_SAMPLE_RATE_HZ;
use crate::backend::AsrModel;
use crate::diarization::{
    AudioSourceOptions, BackendSpeakerId, DiarizationMode, DiarizationOutput, Diarizer,
    DiarizerFactory,
};
use crate::event::{SegmentId, SpeakerActivity, SpeakerId, TranscriptEvent, TranscriptSegment};
use crate::vad::{SpeechChunk, VadFactory, VadGate, duration_from_samples};
use crate::{AudioSourceId, Error, Result};
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;

/// Join handle that completes after both session workers have stopped.
pub type TranscriptionWorker = JoinHandle<()>;

/// A point-in-time snapshot of transcript output delivery statistics.
///
/// All speaker activity, transcript segments, terminal errors, and
/// end-of-stream markers count as events. Counters saturate instead of wrapping
/// when their numeric range is exhausted.
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
/// The receiver also implements [`futures_core::Stream`] with the same items as
/// [`Self::recv`].
pub struct TranscriptEventReceiver {
    inner: mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
    monitor: OutputMonitor,
    cancelled: Arc<AtomicBool>,
    commands: mpsc::WeakUnboundedSender<SessionCommand>,
}

impl TranscriptEventReceiver {
    fn new(
        inner: mpsc::UnboundedReceiver<Result<TranscriptEvent>>,
        state: Arc<OutputState>,
        cancelled: Arc<AtomicBool>,
        commands: mpsc::WeakUnboundedSender<SessionCommand>,
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
            let _ = commands.send(SessionCommand::Cancel);
        }
    }
}

impl futures_core::Stream for TranscriptEventReceiver {
    type Item = Result<TranscriptEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().poll_recv(cx)
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

/// FIFO command sender with bounded capacity for queued audio only.
///
/// Control commands bypass audio backpressure so dropping an input can always
/// queue its close behind that input's previously accepted audio.
#[derive(Clone, Debug)]
pub(crate) struct CommandSender {
    inner: mpsc::UnboundedSender<SessionCommand>,
    audio_capacity: Arc<Semaphore>,
    runtime: Handle,
}

pub(crate) fn command_channel(
    audio_capacity: usize,
) -> (CommandSender, mpsc::UnboundedReceiver<SessionCommand>) {
    let (inner, receiver) = mpsc::unbounded_channel();
    (
        CommandSender {
            inner,
            audio_capacity: Arc::new(Semaphore::new(audio_capacity)),
            runtime: Handle::current(),
        },
        receiver,
    )
}

impl CommandSender {
    fn send(&self, command: SessionCommand) -> Result<()> {
        self.inner.send(command).map_err(|_| Error::StreamClosed)
    }

    async fn send_audio(
        &self,
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
    ) -> Result<()> {
        let capacity_permit = Arc::clone(&self.audio_capacity)
            .acquire_owned()
            .await
            .map_err(|_| Error::StreamClosed)?;
        self.send(SessionCommand::Audio {
            source_id,
            timestamp,
            samples,
            capacity_permit,
        })
    }

    fn blocking_send_audio(
        &self,
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
    ) -> Result<()> {
        let capacity_permit = self
            .runtime
            .block_on(Arc::clone(&self.audio_capacity).acquire_owned())
            .map_err(|_| Error::StreamClosed)?;
        self.send(SessionCommand::Audio {
            source_id,
            timestamp,
            samples,
            capacity_permit,
        })
    }

    pub(crate) fn downgrade(&self) -> mpsc::WeakUnboundedSender<SessionCommand> {
        self.inner.downgrade()
    }

    pub(crate) fn internal_sender(&self) -> mpsc::UnboundedSender<SessionCommand> {
        self.inner.clone()
    }
}

/// Input handle for one registered audio source.
///
/// The handle is intentionally not cloneable so each source has one explicit
/// owner and one unambiguous end-of-stream operation.
/// Dropping the handle queues an ordered non-blocking close request. Call
/// `close` or `blocking_close` explicitly to wait for buffered output and
/// observe any close error.
#[derive(Debug)]
pub struct AudioInput {
    source_id: AudioSourceId,
    commands: CommandSender,
    cancelled: Arc<AtomicBool>,
    closed: bool,
}

impl AudioInput {
    pub(crate) fn new(
        source_id: AudioSourceId,
        commands: CommandSender,
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
    ///
    /// `Ok(())` means the chunk was accepted by the session input queue. It
    /// does not mean VAD or transcription has completed. Processing failures
    /// after acceptance are emitted through [`TranscriptionSession::events`].
    pub async fn send(&self, samples: Vec<f32>) -> Result<()> {
        self.send_command(None, samples).await
    }

    /// Send an f32 mono 16 kHz chunk anchored to the session timeline.
    ///
    /// The timestamp is rounded down to the nearest 16 kHz sample boundary.
    /// The first explicit timestamp anchors this source; later explicit
    /// timestamps must equal the position implied by all previously sent
    /// samples after the same quantization.
    ///
    /// As with [`Self::send`], `Ok(())` only confirms that the chunk was
    /// accepted by the session input queue. Timestamp validation and
    /// processing failures after acceptance are emitted as terminal errors
    /// through [`TranscriptionSession::events`].
    pub async fn send_at(&self, timestamp: Duration, samples: Vec<f32>) -> Result<()> {
        self.send_command(Some(timestamp), samples).await
    }

    async fn send_command(&self, timestamp: Option<Duration>, samples: Vec<f32>) -> Result<()> {
        self.ensure_active()?;
        self.commands
            .send_audio(self.source_id, timestamp, samples)
            .await?;
        self.ensure_active()
    }

    /// Send one f32 mono 16 kHz PCM chunk from a non-async capture thread.
    ///
    /// Like [`Self::send`], `Ok(())` only confirms that the chunk was accepted
    /// by the session input queue. Processing results and failures are
    /// delivered through [`TranscriptionSession::events`].
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
    /// [`Self::send_at`]. `Ok(())` only confirms that the chunk was accepted by
    /// the session input queue.
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
            .blocking_send_audio(self.source_id, timestamp, samples)?;
        self.ensure_active()
    }

    /// Finish and release this source after emitting any buffered segment.
    ///
    pub async fn close(mut self) -> Result<()> {
        self.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands.send(SessionCommand::CloseSource {
            source_id: self.source_id,
            reply: Some(reply_tx),
        })?;
        self.closed = true;
        reply_rx.await.map_err(|_| Error::StreamClosed)?
    }

    /// Finish and release this source from a non-async capture thread.
    ///
    /// # Panics
    ///
    /// Panics when called from an asynchronous execution context.
    pub fn blocking_close(mut self) -> Result<()> {
        self.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands.send(SessionCommand::CloseSource {
            source_id: self.source_id,
            reply: Some(reply_tx),
        })?;
        self.closed = true;
        reply_rx.blocking_recv().map_err(|_| Error::StreamClosed)?
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
            let _ = self.commands.send(SessionCommand::CloseSource {
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
        self.open_source_with_options(AudioSourceOptions::OFF).await
    }

    /// Register an additional audio source with fixed lifetime options.
    ///
    /// Enabling diarization may lazily load the configured model. If loading
    /// fails, only this source creation is rejected and a later call may retry.
    pub async fn open_source_with_options(
        &self,
        options: AudioSourceOptions,
    ) -> Result<AudioInput> {
        self.input.ensure_active()?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.input.commands.send(SessionCommand::OpenSource {
            options,
            reply: reply_tx,
        })?;
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
        options: AudioSourceOptions,
        reply: oneshot::Sender<Result<AudioSourceId>>,
    },
    Audio {
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
        capacity_permit: OwnedSemaphorePermit,
    },
    CloseSource {
        source_id: AudioSourceId,
        reply: Option<oneshot::Sender<Result<()>>>,
    },
    DiarizedJob {
        job: TranscriptionJob,
        capacity_permit: OwnedSemaphorePermit,
    },
    SpeakerActivity(SpeakerActivity),
    DiarizedSourceClosed {
        source_id: AudioSourceId,
        result: Result<()>,
    },
    DiarizationFailed(Error),
}

pub(crate) enum DiarizationCommand {
    OpenSource {
        source_id: AudioSourceId,
        reply: std_mpsc::Sender<Result<()>>,
    },
    Audio {
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
        capacity_permit: OwnedSemaphorePermit,
    },
    CloseSource {
        source_id: AudioSourceId,
    },
    Cancel,
}

#[derive(Debug)]
pub(crate) struct TranscriptionJob {
    source_id: AudioSourceId,
    speaker_id: Option<SpeakerId>,
    start_sample: u64,
    end_sample: u64,
    samples: Vec<f32>,
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
    commands: mpsc::WeakUnboundedSender<SessionCommand>,
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

pub(crate) struct TranscriptionWorkerParams {
    pub(crate) max_sources: usize,
    pub(crate) model: Box<dyn AsrModel>,
    pub(crate) primary_vad: Box<dyn VadGate>,
    pub(crate) vad_factory: Box<dyn VadFactory>,
    pub(crate) primary_options: AudioSourceOptions,
    pub(crate) diarization_tx: mpsc::UnboundedSender<DiarizationCommand>,
    pub(crate) command_rx: mpsc::UnboundedReceiver<SessionCommand>,
    pub(crate) event_tx: EventSender,
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) startup_reply: std_mpsc::Sender<Result<()>>,
}

pub(crate) fn run_transcription_worker(params: TranscriptionWorkerParams) {
    let TranscriptionWorkerParams {
        max_sources,
        mut model,
        primary_vad,
        mut vad_factory,
        primary_options,
        diarization_tx,
        mut command_rx,
        event_tx,
        cancelled,
        startup_reply,
    } = params;
    let _diarization_shutdown = DiarizationShutdown(diarization_tx.clone());
    let mut next_segment_id = 0u64;
    let mut next_source_id = 1u64;
    let primary_state = match primary_options.diarization {
        DiarizationMode::Off => Ok(SourceState::Direct(DirectSourceState::new(primary_vad))),
        DiarizationMode::On => open_diarization_source(&diarization_tx, AudioSourceId::PRIMARY)
            .map(|()| SourceState::Diarized {
                closing: false,
                close_reply: None,
            }),
    };
    let mut sources = match primary_state {
        Ok(state) => {
            let _ = startup_reply.send(Ok(()));
            vec![(AudioSourceId::PRIMARY, state)]
        }
        Err(err) => {
            let _ = startup_reply.send(Err(err));
            return;
        }
    };

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
            SessionCommand::OpenSource { options, reply } => {
                if sources.len() >= max_sources {
                    let _ = reply.send(Err(Error::SourceLimit { max_sources }));
                    continue;
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

                let source =
                    match options.diarization {
                        DiarizationMode::Off => vad_factory
                            .create()
                            .map(|vad| SourceState::Direct(DirectSourceState::new(vad))),
                        DiarizationMode::On => open_diarization_source(&diarization_tx, source_id)
                            .map(|()| SourceState::Diarized {
                                closing: false,
                                close_reply: None,
                            }),
                    };
                let mut source = match source {
                    Ok(source) => source,
                    Err(err) => {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                };
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
                if reply.send(Ok(source_id)).is_err() {
                    if let SourceState::Diarized { closing, .. } = &mut source {
                        *closing = true;
                        sources.push((source_id, source));
                        if diarization_tx
                            .send(DiarizationCommand::CloseSource { source_id })
                            .is_err()
                        {
                            fail(&event_tx, Error::StreamClosed);
                            return;
                        }
                    }
                    continue;
                }
                sources.push((source_id, source));
            }
            SessionCommand::Audio {
                source_id,
                timestamp,
                samples,
                capacity_permit,
            } => {
                let Some(source) = sources
                    .iter_mut()
                    .find(|(id, _)| *id == source_id)
                    .map(|(_, source)| source)
                else {
                    fail(&event_tx, Error::SourceNotFound { source_id });
                    return;
                };
                let result = match source {
                    SourceState::Direct(source) => {
                        drop(capacity_permit);
                        let mut sink = EventSink {
                            event_tx: &event_tx,
                            next_segment_id: &mut next_segment_id,
                            cancelled: cancelled.as_ref(),
                        };
                        process_chunk(
                            model.as_mut(),
                            source,
                            &mut sink,
                            source_id,
                            timestamp,
                            samples,
                        )
                    }
                    SourceState::Diarized { closing, .. } => {
                        if *closing {
                            Err(Error::SourceNotFound { source_id })
                        } else {
                            diarization_tx
                                .send(DiarizationCommand::Audio {
                                    source_id,
                                    timestamp,
                                    samples,
                                    capacity_permit,
                                })
                                .map_err(|_| Error::StreamClosed)
                        }
                    }
                };
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
                match &mut sources[position].1 {
                    SourceState::Direct(_) => {
                        let (_, SourceState::Direct(source)) = sources.remove(position) else {
                            unreachable!("source variant changed while closing")
                        };
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
                    SourceState::Diarized {
                        closing,
                        close_reply,
                    } => {
                        if *closing {
                            if let Some(reply) = reply {
                                let _ = reply.send(Err(Error::SourceNotFound { source_id }));
                            }
                            continue;
                        }
                        *closing = true;
                        *close_reply = reply;
                        if diarization_tx
                            .send(DiarizationCommand::CloseSource { source_id })
                            .is_err()
                        {
                            fail(&event_tx, Error::StreamClosed);
                            return;
                        }
                    }
                }
            }
            SessionCommand::DiarizedJob {
                job,
                capacity_permit,
            } => {
                let active = sources.iter().any(|(source_id, source)| {
                    *source_id == job.source_id && matches!(source, SourceState::Diarized { .. })
                });
                if !active {
                    fail(
                        &event_tx,
                        Error::SourceNotFound {
                            source_id: job.source_id,
                        },
                    );
                    return;
                }
                let mut sink = EventSink {
                    event_tx: &event_tx,
                    next_segment_id: &mut next_segment_id,
                    cancelled: cancelled.as_ref(),
                };
                let result = handle_transcription_job(model.as_mut(), &mut sink, job);
                drop(capacity_permit);
                if cancelled.load(Ordering::Acquire) {
                    return;
                }
                if let Err(err) = result {
                    fail(&event_tx, err);
                    return;
                }
            }
            SessionCommand::SpeakerActivity(activity) => {
                if event_tx
                    .send(Ok(TranscriptEvent::SpeakerActivity(activity)))
                    .is_err()
                {
                    return;
                }
            }
            SessionCommand::DiarizedSourceClosed { source_id, result } => {
                let Some(position) = sources.iter().position(|(id, _)| *id == source_id) else {
                    fail(&event_tx, Error::SourceNotFound { source_id });
                    return;
                };
                let (_, source) = sources.remove(position);
                let SourceState::Diarized { close_reply, .. } = source else {
                    fail(&event_tx, Error::SourceNotFound { source_id });
                    return;
                };
                if let Some(reply) = close_reply {
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
            SessionCommand::DiarizationFailed(err) => {
                fail(&event_tx, err);
                return;
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
        let SourceState::Direct(source) = source else {
            return;
        };
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

struct DiarizationShutdown(mpsc::UnboundedSender<DiarizationCommand>);

impl Drop for DiarizationShutdown {
    fn drop(&mut self) {
        let _ = self.0.send(DiarizationCommand::Cancel);
    }
}

enum SourceState {
    Direct(DirectSourceState),
    Diarized {
        closing: bool,
        close_reply: Option<oneshot::Sender<Result<()>>>,
    },
}

struct DirectSourceState {
    vad: Box<dyn VadGate>,
    timeline: SourceTimeline,
}

impl DirectSourceState {
    fn new(vad: Box<dyn VadGate>) -> Self {
        Self {
            vad,
            timeline: SourceTimeline::default(),
        }
    }
}

#[derive(Default)]
struct SourceTimeline {
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
    timeline: &mut SourceTimeline,
    timestamp: Option<Duration>,
) -> Result<u64> {
    let explicit_sample = timestamp
        .map(|timestamp| session_sample_from_duration(source_id, timestamp))
        .transpose()?;

    match timeline.timeline_offset_sample {
        Some(offset) => {
            if let Some(actual_sample) = explicit_sample {
                let expected_sample =
                    timeline_sample(source_id, offset, timeline.next_input_sample)?;
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
            timeline.timeline_offset_sample = Some(offset);
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
    source: &mut DirectSourceState,
    sink: &mut EventSink<'_>,
    source_id: AudioSourceId,
    timestamp: Option<Duration>,
    samples: Vec<f32>,
) -> Result<()> {
    let timeline_offset_sample =
        resolve_timeline_offset(source_id, &mut source.timeline, timestamp)?;
    let start_sample = source.timeline.next_input_sample;
    source.timeline.next_input_sample = source
        .timeline
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
    mut source: DirectSourceState,
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
    let timeline_offset_sample = source.timeline.timeline_offset_sample.unwrap_or(0);
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
        let job = TranscriptionJob {
            source_id,
            speaker_id: None,
            start_sample: timeline_sample(source_id, timeline_offset_sample, speech.start_sample)?,
            end_sample: timeline_sample(source_id, timeline_offset_sample, speech.end_sample)?,
            samples: speech.samples,
        };
        handle_transcription_job(model, sink, job)?;
    }
    Ok(())
}

fn handle_transcription_job(
    model: &mut dyn AsrModel,
    sink: &mut EventSink<'_>,
    mut job: TranscriptionJob,
) -> Result<()> {
    if job.samples.is_empty() || sink.cancelled.load(Ordering::Acquire) {
        return Ok(());
    }
    let started = Instant::now();
    let text = model.transcribe(std::mem::take(&mut job.samples))?;
    if sink.cancelled.load(Ordering::Acquire) || text.trim().is_empty() {
        return Ok(());
    }
    send_transcript(sink, job, text, started.elapsed())
}

fn send_transcript(
    sink: &mut EventSink<'_>,
    job: TranscriptionJob,
    text: String,
    inference_duration: Duration,
) -> Result<()> {
    let id = SegmentId::new(*sink.next_segment_id);
    *sink.next_segment_id = sink.next_segment_id.saturating_add(1);
    sink.event_tx
        .send(Ok(TranscriptEvent::Segment(TranscriptSegment {
            id,
            source_id: job.source_id,
            speaker_id: job.speaker_id,
            text,
            start: duration_from_samples(job.start_sample),
            end: duration_from_samples(job.end_sample),
            inference_duration,
            is_final: true,
        })))
}

fn open_diarization_source(
    diarization_tx: &mpsc::UnboundedSender<DiarizationCommand>,
    source_id: AudioSourceId,
) -> Result<()> {
    let (reply_tx, reply_rx) = std_mpsc::channel();
    diarization_tx
        .send(DiarizationCommand::OpenSource {
            source_id,
            reply: reply_tx,
        })
        .map_err(|_| Error::StreamClosed)?;
    reply_rx.recv().map_err(|_| Error::StreamClosed)?
}

pub(crate) fn run_diarization_worker(
    factory: Box<dyn DiarizerFactory>,
    mut command_rx: mpsc::UnboundedReceiver<DiarizationCommand>,
    session_tx: mpsc::UnboundedSender<SessionCommand>,
    job_capacity: Arc<Semaphore>,
    runtime: Handle,
    cancelled: Arc<AtomicBool>,
) {
    let mut state = DiarizationWorkerState {
        factory,
        diarizer: None,
        sources: HashMap::new(),
        speaker_ids: HashMap::new(),
        next_speaker_id: 0,
        session_tx,
        job_capacity,
        runtime,
        cancelled,
    };

    while let Some(command) = command_rx.blocking_recv() {
        match command {
            DiarizationCommand::Cancel => return,
            DiarizationCommand::OpenSource { source_id, reply } => {
                let _ = reply.send(state.open_source(source_id));
            }
            DiarizationCommand::Audio {
                source_id,
                timestamp,
                samples,
                capacity_permit,
            } => {
                // This permit bounds commands waiting to be consumed, not PCM
                // retained for backend lookahead. Keeping it past this point
                // can prevent the very input needed to advance the watermark.
                drop(capacity_permit);
                let result = state.process_audio(source_id, timestamp, samples);
                if let Err(err) = result {
                    let _ = state
                        .session_tx
                        .send(SessionCommand::DiarizationFailed(err));
                    return;
                }
            }
            DiarizationCommand::CloseSource { source_id } => {
                let result = state.finish_source(source_id);
                let failed = result.is_err();
                if state
                    .session_tx
                    .send(SessionCommand::DiarizedSourceClosed { source_id, result })
                    .is_err()
                {
                    return;
                }
                if failed {
                    return;
                }
            }
        }
    }
}

struct DiarizationWorkerState {
    factory: Box<dyn DiarizerFactory>,
    diarizer: Option<Box<dyn Diarizer>>,
    sources: HashMap<AudioSourceId, DiarizationSourceState>,
    speaker_ids: HashMap<(AudioSourceId, BackendSpeakerId), SpeakerId>,
    next_speaker_id: u64,
    session_tx: mpsc::UnboundedSender<SessionCommand>,
    job_capacity: Arc<Semaphore>,
    runtime: Handle,
    cancelled: Arc<AtomicBool>,
}

impl DiarizationWorkerState {
    fn open_source(&mut self, source_id: AudioSourceId) -> Result<()> {
        if self.sources.contains_key(&source_id) {
            return Err(Error::InvalidConfig(format!(
                "diarization source {} is already active",
                source_id.get()
            )));
        }
        if self.diarizer.is_none() {
            self.diarizer = Some(self.factory.create()?);
        }
        self.diarizer
            .as_mut()
            .expect("diarizer initialized above")
            .open_source(source_id)?;
        self.sources
            .insert(source_id, DiarizationSourceState::default());
        Ok(())
    }

    fn process_audio(
        &mut self,
        source_id: AudioSourceId,
        timestamp: Option<Duration>,
        samples: Vec<f32>,
    ) -> Result<()> {
        let source = self
            .sources
            .get_mut(&source_id)
            .ok_or(Error::SourceNotFound { source_id })?;
        let timeline_offset = resolve_timeline_offset(source_id, &mut source.timeline, timestamp)?;
        let start_sample = source.timeline.next_input_sample;
        source.timeline.next_input_sample = source
            .timeline
            .next_input_sample
            .saturating_add(samples.len() as u64);
        let output = self
            .diarizer
            .as_deref_mut()
            .ok_or_else(|| Error::Backend("diarization model is not initialized".to_string()))?
            .push(source_id, &samples, start_sample, &self.cancelled)?;
        source.buffer.append(start_sample, samples)?;
        self.process_output(source_id, timeline_offset, output)
    }

    fn finish_source(&mut self, source_id: AudioSourceId) -> Result<()> {
        let source = self
            .sources
            .get(&source_id)
            .ok_or(Error::SourceNotFound { source_id })?;
        let timeline_offset = source.timeline.timeline_offset_sample.unwrap_or(0);
        let output = self
            .diarizer
            .as_deref_mut()
            .ok_or_else(|| Error::Backend("diarization model is not initialized".to_string()))?
            .finish(source_id, &self.cancelled)?;
        self.process_output(source_id, timeline_offset, output)?;

        let source = self
            .sources
            .get(&source_id)
            .expect("source remains active until its flush is validated");
        if source.finalized_until != source.timeline.next_input_sample {
            return Err(invalid_diarization_output(format!(
                "flush for source {} finalized through sample {}, expected {}",
                source_id.get(),
                source.finalized_until,
                source.timeline.next_input_sample
            )));
        }
        self.sources.remove(&source_id);
        self.speaker_ids
            .retain(|(registered_source, _), _| *registered_source != source_id);
        Ok(())
    }

    fn process_output(
        &mut self,
        source_id: AudioSourceId,
        timeline_offset: u64,
        output: DiarizationOutput,
    ) -> Result<()> {
        let max_retained_samples = self
            .diarizer
            .as_deref()
            .ok_or_else(|| Error::Backend("diarization model is not initialized".to_string()))?
            .max_retained_samples();
        let retained_samples = self.sources.values().try_fold(0usize, |total, source| {
            total.checked_add(source.buffer.len()).ok_or_else(|| {
                invalid_diarization_output("retained PCM sample count overflowed".to_string())
            })
        })?;
        let Self {
            sources,
            speaker_ids,
            next_speaker_id,
            session_tx,
            job_capacity,
            runtime,
            ..
        } = self;
        let source = sources
            .get_mut(&source_id)
            .ok_or(Error::SourceNotFound { source_id })?;
        if output.finalized_until < source.finalized_until
            || output.finalized_until > source.timeline.next_input_sample
        {
            return Err(invalid_diarization_output(format!(
                "source {} returned invalid watermark {} after {} with {} input samples",
                source_id.get(),
                output.finalized_until,
                source.finalized_until,
                source.timeline.next_input_sample
            )));
        }

        let newly_finalized = usize::try_from(output.finalized_until - source.finalized_until)
            .map_err(|_| {
                invalid_diarization_output(
                    "finalized PCM sample count does not fit in memory".to_string(),
                )
            })?;
        let retained_after = retained_samples
            .checked_sub(newly_finalized)
            .ok_or_else(|| {
                invalid_diarization_output(format!(
                    "source {} finalized more PCM than the session retained",
                    source_id.get()
                ))
            })?;
        if retained_after > max_retained_samples {
            return Err(invalid_diarization_output(format!(
                "backend retained {retained_after} PCM samples, exceeding its declared limit of {max_retained_samples}"
            )));
        }

        let mut previous_end = source.finalized_until;
        for region in output.regions {
            if region.start_sample < previous_end
                || region.start_sample >= region.end_sample
                || region.end_sample > output.finalized_until
            {
                return Err(invalid_diarization_output(format!(
                    "source {} returned invalid finalized region {}..{}",
                    source_id.get(),
                    region.start_sample,
                    region.end_sample
                )));
            }
            previous_end = region.end_sample;
            if region.speakers.is_empty() {
                continue;
            }

            let samples = source
                .buffer
                .copy(region.start_sample, region.end_sample)
                .ok_or_else(|| {
                    invalid_diarization_output(format!(
                        "source {} region {}..{} is outside retained PCM",
                        source_id.get(),
                        region.start_sample,
                        region.end_sample
                    ))
                })?;
            let start_sample = timeline_sample(source_id, timeline_offset, region.start_sample)?;
            let end_sample = timeline_sample(source_id, timeline_offset, region.end_sample)?;
            let mut activity_speakers = Vec::with_capacity(region.speakers.len());
            for backend_speaker in region.speakers {
                let key = (source_id, backend_speaker);
                let speaker_id = match speaker_ids.get(&key) {
                    Some(id) => *id,
                    None => {
                        let id = SpeakerId::new(*next_speaker_id);
                        *next_speaker_id = next_speaker_id.checked_add(1).ok_or_else(|| {
                            Error::InvalidConfig("speaker identifier space exhausted".to_string())
                        })?;
                        speaker_ids.insert(key, id);
                        id
                    }
                };
                activity_speakers.push(speaker_id);
            }
            for speaker_id in &activity_speakers {
                session_tx
                    .send(SessionCommand::SpeakerActivity(SpeakerActivity {
                        source_id,
                        speaker_id: *speaker_id,
                        start: duration_from_samples(start_sample),
                        end: duration_from_samples(end_sample),
                    }))
                    .map_err(|_| Error::StreamClosed)?;
            }
            let speaker_id = if activity_speakers.len() == 1 {
                Some(activity_speakers[0])
            } else {
                None
            };
            let capacity_permit = runtime
                .block_on(Arc::clone(job_capacity).acquire_owned())
                .map_err(|_| Error::StreamClosed)?;
            let job = TranscriptionJob {
                source_id,
                speaker_id,
                start_sample,
                end_sample,
                samples,
            };
            session_tx
                .send(SessionCommand::DiarizedJob {
                    job,
                    capacity_permit,
                })
                .map_err(|_| Error::StreamClosed)?;
        }

        source.buffer.discard_before(output.finalized_until);
        source.finalized_until = output.finalized_until;
        Ok(())
    }
}

fn invalid_diarization_output(message: String) -> Error {
    Error::Backend(format!("invalid diarization output: {message}"))
}

#[derive(Default)]
struct DiarizationSourceState {
    timeline: SourceTimeline,
    finalized_until: u64,
    buffer: RetainedPcm,
}

#[derive(Default)]
struct RetainedPcm {
    chunks: VecDeque<RetainedChunk>,
    sample_count: usize,
}

struct RetainedChunk {
    start_sample: u64,
    samples: Vec<f32>,
}

impl RetainedPcm {
    fn append(&mut self, start_sample: u64, samples: Vec<f32>) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let sample_count = self
            .sample_count
            .checked_add(samples.len())
            .ok_or_else(|| {
                invalid_diarization_output("retained PCM sample count overflowed".to_string())
            })?;
        if let Some(last) = self.chunks.back() {
            let expected = last.start_sample.saturating_add(last.samples.len() as u64);
            if start_sample != expected {
                return Err(invalid_diarization_output(format!(
                    "retained PCM starts at {start_sample}, expected {expected}"
                )));
            }
        }
        self.chunks.push_back(RetainedChunk {
            start_sample,
            samples,
        });
        self.sample_count = sample_count;
        Ok(())
    }

    fn len(&self) -> usize {
        self.sample_count
    }

    fn copy(&self, start_sample: u64, end_sample: u64) -> Option<Vec<f32>> {
        let expected_len = usize::try_from(end_sample.checked_sub(start_sample)?).ok()?;
        let mut samples = Vec::with_capacity(expected_len);
        for chunk in &self.chunks {
            let chunk_end = chunk
                .start_sample
                .saturating_add(chunk.samples.len() as u64);
            let copy_start = start_sample.max(chunk.start_sample);
            let copy_end = end_sample.min(chunk_end);
            if copy_start >= copy_end {
                continue;
            }
            let relative_start = usize::try_from(copy_start - chunk.start_sample).ok()?;
            let relative_end = usize::try_from(copy_end - chunk.start_sample).ok()?;
            samples.extend_from_slice(&chunk.samples[relative_start..relative_end]);
        }
        (samples.len() == expected_len).then_some(samples)
    }

    fn discard_before(&mut self, sample: u64) {
        while self.chunks.front().is_some_and(|chunk| {
            chunk
                .start_sample
                .saturating_add(chunk.samples.len() as u64)
                <= sample
        }) {
            let removed = self
                .chunks
                .pop_front()
                .expect("front chunk exists while discarding retained PCM");
            self.sample_count = self.sample_count.saturating_sub(removed.samples.len());
        }
        let Some(chunk) = self.chunks.front_mut() else {
            return;
        };
        if sample <= chunk.start_sample {
            return;
        }
        let count = usize::try_from(sample - chunk.start_sample)
            .unwrap_or(usize::MAX)
            .min(chunk.samples.len());
        chunk.samples.drain(..count);
        chunk.start_sample = chunk.start_sample.saturating_add(count as u64);
        self.sample_count = self.sample_count.saturating_sub(count);
    }
}

fn fail(event_tx: &EventSender, err: Error) {
    let _ = event_tx.send(Err(err));
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::future::{Future, poll_fn};

    fn test_output_channel() -> (
        EventSender,
        TranscriptEventReceiver,
        OutputMonitor,
        mpsc::UnboundedSender<SessionCommand>,
    ) {
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
        let (event_tx, events, _cancelled) = output_channel(command_tx.downgrade());
        let monitor = events.monitor();
        (event_tx, events, monitor, command_tx)
    }

    fn end_event() -> Result<TranscriptEvent> {
        Ok(TranscriptEvent::EndOfStream)
    }

    #[tokio::test]
    async fn receiver_stream_tracks_received_events_and_closure() {
        let (event_tx, mut events, monitor, _command_tx) = test_output_channel();
        event_tx.send(end_event()).unwrap();
        drop(event_tx);

        assert!(events.next().await.unwrap().is_ok());
        assert_eq!(monitor.metrics().received_events, 1);
        assert!(events.next().await.is_none());
        assert_eq!(monitor.metrics().received_events, 1);
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
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
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
    async fn input_close_uses_worker_reply_as_its_linearization_point() {
        let (command_tx, mut command_rx) = command_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let input = AudioInput::new(AudioSourceId::PRIMARY, command_tx, Arc::clone(&cancelled));

        let responder = tokio::spawn(async move {
            let SessionCommand::CloseSource { reply, .. } = command_rx.recv().await.unwrap() else {
                panic!("expected close command");
            };
            cancelled.store(true, Ordering::Release);
            reply.unwrap().send(Ok(())).unwrap();
        });

        assert_eq!(input.close().await, Ok(()));
        responder.await.unwrap();
    }

    #[tokio::test]
    async fn audio_send_waits_for_available_capacity() {
        let (command_tx, mut command_rx) = command_channel(1);
        command_tx
            .send_audio(AudioSourceId::PRIMARY, None, vec![0.1])
            .await
            .unwrap();
        let mut second_send =
            Box::pin(command_tx.send_audio(AudioSourceId::PRIMARY, None, vec![0.2]));
        poll_fn(|cx| {
            assert!(second_send.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;

        let SessionCommand::Audio {
            capacity_permit,
            samples,
            ..
        } = command_rx.recv().await.unwrap()
        else {
            panic!("expected first audio command");
        };
        assert_eq!(samples, vec![0.1]);
        drop(capacity_permit);

        second_send.await.unwrap();
        let SessionCommand::Audio {
            capacity_permit,
            samples,
            ..
        } = command_rx.recv().await.unwrap()
        else {
            panic!("expected second audio command");
        };
        assert_eq!(samples, vec![0.2]);
        drop(capacity_permit);
    }

    #[tokio::test]
    async fn blocking_audio_send_works_from_capture_thread() {
        let (command_tx, mut command_rx) = command_channel(1);
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let sender = std::thread::spawn(move || {
            let result = command_tx.blocking_send_audio(AudioSourceId::PRIMARY, None, vec![0.1]);
            result_tx.send(result).unwrap();
        });

        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            Ok(())
        );
        let SessionCommand::Audio {
            capacity_permit,
            samples,
            ..
        } = command_rx.try_recv().unwrap()
        else {
            panic!("expected audio command");
        };
        assert_eq!(samples, vec![0.1]);
        drop(capacity_permit);
        sender.join().unwrap();
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
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
        assert_eq!(command_tx.strong_count(), 1);

        let (_event_tx, events, _cancelled) = output_channel(command_tx.downgrade());
        let monitor = events.monitor();
        assert_eq!(command_tx.strong_count(), 1);

        drop(events);
        drop(monitor);
        assert_eq!(command_tx.strong_count(), 1);
    }
}
