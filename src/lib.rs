//! Parakeet-family on-device transcription for desktop applications.
//!
//! The crate owns the PCM streaming API, VAD gating, timestamp accounting, and
//! Parakeet TDT inference. Audio capture, resampling, downmix, and model
//! download stay with the application.
//!
//! # Example
//!
//! ```no_run
//! use yamabiko_asr::{Error, TranscriptEvent, Transcriber};
//!
//! # async fn transcribe() -> yamabiko_asr::Result<()> {
//! let transcriber = Transcriber::builder("path/to/model").build()?;
//! let (input, mut events, worker) = transcriber.start().into_parts();
//!
//! input.send(vec![0.0; 1_600]).await?;
//! input.close().await?;
//!
//! while let Some(event) = events.recv().await {
//!     match event? {
//!         TranscriptEvent::Segment(segment) => println!("{}", segment.text),
//!         TranscriptEvent::EndOfStream => break,
//!         _ => {}
//!     }
//! }
//! worker.await.map_err(|_| Error::StreamClosed)?;
//! # Ok(())
//! # }
//! ```

mod backend;
mod builder;
mod config;
mod error;
mod event;
mod session;
mod tdt;
mod vad;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

pub use builder::TranscriberBuilder;
pub(crate) use config::TranscriberConfig;
pub use config::{AudioSourceId, Device, PCM_SAMPLE_RATE_HZ};
pub use error::{Error, Result};
pub use event::{SegmentId, SpeakerId, TranscriptEvent, TranscriptSegment};
pub use session::{
    AudioInput, OutputMetrics, OutputMonitor, TranscriptEventReceiver, TranscriptionSession,
    TranscriptionWorker,
};

use session::{SessionCommand, output_channel, run_transcription_worker};
use tdt::ParakeetTdtModel;
use vad::{SileroVadFactory, VadFactory};

/// A transcription engine backed by one loaded Parakeet ASR model.
///
/// Multiple audio sources share this model while keeping independent VAD,
/// buffering, and source-local timeline state. Sources are scheduled
/// independently and are not mixed before transcription.
pub struct Transcriber {
    input_capacity: usize,
    max_sources: usize,
    model: Box<dyn backend::AsrModel>,
    vad: Box<dyn vad::VadGate>,
    vad_factory: Box<dyn VadFactory>,
}

impl Transcriber {
    /// Load the ASR model and VAD backend from a validated configuration.
    ///
    /// Model loading is synchronous and may take long enough that GUI
    /// applications should call this away from their UI thread.
    pub(crate) fn new(config: TranscriberConfig) -> Result<Self> {
        let vad_options = config.validate()?;
        let model: Box<dyn backend::AsrModel> =
            Box::new(ParakeetTdtModel::load(&config.model_dir, config.device)?);
        let mut vad_factory = SileroVadFactory::new(vad_options)?;
        let vad = vad_factory.create()?;
        Ok(Self {
            input_capacity: config.input_capacity,
            max_sources: config.max_sources,
            model,
            vad,
            vad_factory: Box::new(vad_factory),
        })
    }

    /// Start the Tokio-facing streaming input/output API.
    ///
    /// The worker runs on Tokio's blocking pool because ONNX inference is
    /// synchronous. The input command channel is bounded to provide natural
    /// backpressure; transcript events use an unbounded channel so closing an
    /// input never depends on draining output concurrently.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime.
    pub fn start(self) -> TranscriptionSession {
        let (command_tx, command_rx) =
            tokio::sync::mpsc::channel::<SessionCommand>(self.input_capacity);
        let (event_tx, events, cancelled) = output_channel(command_tx.downgrade());
        let worker_cancelled = std::sync::Arc::clone(&cancelled);
        let worker = tokio::task::spawn_blocking(move || {
            run_transcription_worker(
                self.max_sources,
                self.model,
                self.vad,
                self.vad_factory,
                command_rx,
                event_tx,
                worker_cancelled,
            );
        });

        TranscriptionSession {
            input: AudioInput::new(AudioSourceId::PRIMARY, command_tx, cancelled),
            events,
            worker,
        }
    }

    #[cfg(test)]
    fn from_parts(
        config: TranscriberConfig,
        model: Box<dyn backend::AsrModel>,
        vad: Box<dyn vad::VadGate>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            vad_factory: Box::new(UnavailableVadFactory),
            input_capacity: config.input_capacity,
            max_sources: config.max_sources,
            model,
            vad,
        })
    }
}

#[cfg(test)]
struct UnavailableVadFactory;

#[cfg(test)]
impl VadFactory for UnavailableVadFactory {
    fn create(&mut self) -> Result<Box<dyn vad::VadGate>> {
        Err(Error::Vad("no test VAD configured".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc as std_mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct FakeModel {
        calls: Arc<Mutex<Vec<usize>>>,
        next: usize,
    }

    impl backend::AsrModel for FakeModel {
        fn transcribe(&mut self, samples: Vec<f32>) -> Result<String> {
            self.calls.lock().unwrap().push(samples.len());
            self.next += 1;
            Ok(format!("chunk{}", self.next))
        }
    }

    struct BlockingModel {
        started: std_mpsc::Sender<()>,
        release: std_mpsc::Receiver<()>,
    }

    impl backend::AsrModel for BlockingModel {
        fn transcribe(&mut self, _samples: Vec<f32>) -> Result<String> {
            self.started.send(()).unwrap();
            self.release.recv().unwrap();
            Ok("finished".to_string())
        }
    }

    struct FakeVad {
        chunks: Vec<Vec<vad::SpeechChunk>>,
    }

    impl vad::VadGate for FakeVad {
        fn push(&mut self, _samples: &[f32], _start_sample: u64) -> Result<Vec<vad::SpeechChunk>> {
            if self.chunks.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(self.chunks.remove(0))
            }
        }

        fn finish(&mut self) -> Result<Vec<vad::SpeechChunk>> {
            Ok(Vec::new())
        }
    }

    struct FakeVadFactory {
        vads: Vec<Box<dyn vad::VadGate>>,
    }

    impl vad::VadFactory for FakeVadFactory {
        fn create(&mut self) -> Result<Box<dyn vad::VadGate>> {
            if self.vads.is_empty() {
                Err(Error::Vad("no fake VAD configured".to_string()))
            } else {
                Ok(self.vads.remove(0))
            }
        }
    }

    struct FinishVad {
        chunks: Vec<vad::SpeechChunk>,
    }

    impl vad::VadGate for FinishVad {
        fn push(&mut self, _samples: &[f32], _start_sample: u64) -> Result<Vec<vad::SpeechChunk>> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> Result<Vec<vad::SpeechChunk>> {
            Ok(std::mem::take(&mut self.chunks))
        }
    }

    fn test_config() -> TranscriberConfig {
        TranscriberConfig::new("model-dir")
    }

    #[tokio::test]
    async fn vad_gating_keeps_silence_out_of_backend() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let model = FakeModel {
            calls: Arc::clone(&calls),
            next: 0,
        };
        let vad = FakeVad {
            chunks: vec![
                Vec::new(),
                vec![vad::SpeechChunk {
                    samples: vec![0.2; 4],
                    start_sample: 16_000,
                    end_sample: 16_004,
                }],
            ],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session.input.send(vec![0.0; 4]).await.unwrap();
        session.input.send(vec![0.2; 4]).await.unwrap();
        session.input.close().await.unwrap();

        while let Some(event) = session.events.recv().await {
            if matches!(event.unwrap(), TranscriptEvent::EndOfStream) {
                break;
            }
        }

        assert_eq!(*calls.lock().unwrap(), vec![4]);
    }

    #[tokio::test]
    async fn timestamps_use_session_audio_timeline() {
        let model = FakeModel {
            calls: Arc::new(Mutex::new(Vec::new())),
            next: 0,
        };
        let vad = FakeVad {
            chunks: vec![vec![vad::SpeechChunk {
                samples: vec![0.2; 8],
                start_sample: 16_000,
                end_sample: 24_000,
            }]],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session.input.send(vec![0.2; 8]).await.unwrap();
        session.input.close().await.unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = event else {
            panic!("expected segment event");
        };

        assert_eq!(segment.start, Duration::from_secs(1));
        assert_eq!(segment.end, Duration::from_millis(1500));
        assert!(segment.is_final);
    }

    #[tokio::test]
    async fn stream_emits_final_and_end_events() {
        let model = FakeModel {
            calls: Arc::new(Mutex::new(Vec::new())),
            next: 0,
        };
        let vad = FakeVad {
            chunks: vec![vec![vad::SpeechChunk {
                samples: vec![0.1; 8],
                start_sample: 0,
                end_sample: 8,
            }]],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session.input.send(vec![0.1; 8]).await.unwrap();
        session.input.close().await.unwrap();

        let first = session.events.recv().await.unwrap().unwrap();
        let second = session.events.recv().await.unwrap().unwrap();

        assert!(matches!(
            first,
            TranscriptEvent::Segment(TranscriptSegment { is_final: true, .. })
        ));
        assert!(matches!(second, TranscriptEvent::EndOfStream));
    }

    #[tokio::test]
    async fn session_into_parts_keeps_worker_joinable() {
        let model = FakeModel {
            calls: Arc::new(Mutex::new(Vec::new())),
            next: 0,
        };
        let vad = FakeVad { chunks: Vec::new() };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let (input, mut events, worker) = transcriber.start().into_parts();
        input.close().await.unwrap();

        let event = events.recv().await.unwrap().unwrap();
        assert!(matches!(event, TranscriptEvent::EndOfStream));
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn dropping_last_input_flushes_without_receiver_keeping_commands_alive() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FinishVad {
                chunks: vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }],
            }),
        )
        .unwrap();

        let (input, mut events, worker) = transcriber.start().into_parts();
        let monitor = events.monitor();
        drop(input);

        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        assert_eq!(events.recv().await, None);
        worker.await.unwrap();
        assert_eq!(monitor.metrics().received_events, 2);
    }

    #[tokio::test]
    async fn multiple_sources_share_one_model_and_keep_separate_ids() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let model = FakeModel {
            calls: Arc::clone(&calls),
            next: 0,
        };
        let primary_vad = FakeVad {
            chunks: vec![vec![vad::SpeechChunk {
                samples: vec![0.1; 4],
                start_sample: 0,
                end_sample: 4,
            }]],
        };
        let vad_factory = FakeVadFactory {
            vads: vec![Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.2; 8],
                    start_sample: 0,
                    end_sample: 8,
                }]],
            })],
        };
        let transcriber = Transcriber {
            input_capacity: 32,
            max_sources: 2,
            model: Box::new(model),
            vad: Box::new(primary_vad),
            vad_factory: Box::new(vad_factory),
        };

        let mut session = transcriber.start();
        let system_input = session.open_source().await.unwrap();
        let system_source = system_input.source_id();
        session.input.send(vec![0.1; 4]).await.unwrap();
        system_input
            .send_at(Duration::from_millis(350), vec![0.2; 8])
            .await
            .unwrap();
        system_input.close().await.unwrap();
        session.input.close().await.unwrap();

        let first = session.events.recv().await.unwrap().unwrap();
        let second = session.events.recv().await.unwrap().unwrap();
        let third = session.events.recv().await.unwrap().unwrap();

        let TranscriptEvent::Segment(first) = first else {
            panic!("expected primary segment");
        };
        let TranscriptEvent::Segment(second) = second else {
            panic!("expected system segment");
        };
        assert_eq!(first.id.get(), 0);
        assert_eq!(first.source_id, AudioSourceId::PRIMARY);
        assert_eq!(first.start, Duration::ZERO);
        assert_eq!(second.id.get(), 1);
        assert_eq!(second.source_id, system_source);
        assert_eq!(second.start, Duration::from_millis(350));
        assert!(matches!(third, TranscriptEvent::EndOfStream));
        assert_eq!(*calls.lock().unwrap(), vec![4, 8]);
    }

    #[tokio::test]
    async fn terminal_error_follows_segments_without_end_of_stream() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 160],
                    start_sample: 0,
                    end_sample: 160,
                }]],
            }),
        )
        .unwrap();
        let mut session = transcriber.start();

        session
            .input
            .send_at(Duration::from_millis(100), vec![0.0; 160])
            .await
            .unwrap();
        session
            .input
            .send_at(Duration::from_millis(105), vec![0.0; 160])
            .await
            .unwrap();

        let segment = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = segment else {
            panic!("expected segment before terminal error");
        };
        assert_eq!(segment.start, Duration::from_millis(100));
        assert_eq!(segment.end, Duration::from_millis(110));

        let error = session.events.recv().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            Error::TimestampDiscontinuity {
                source_id: AudioSourceId::PRIMARY,
                expected,
                actual,
            } if expected == Duration::from_millis(110)
                && actual == Duration::from_millis(105)
        ));
        assert_eq!(session.events.recv().await, None);
        session.worker.await.unwrap();
    }

    #[tokio::test]
    async fn anchored_source_continues_on_session_timeline_with_plain_send() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad {
                chunks: vec![
                    Vec::new(),
                    vec![vad::SpeechChunk {
                        samples: vec![0.2; 160],
                        start_sample: 160,
                        end_sample: 320,
                    }],
                ],
            }),
        )
        .unwrap();
        let mut session = transcriber.start();

        session
            .input
            .send_at(Duration::from_millis(100), vec![0.0; 160])
            .await
            .unwrap();
        session.input.send(vec![0.2; 160]).await.unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = event else {
            panic!("expected anchored segment");
        };
        assert_eq!(segment.start, Duration::from_millis(110));
        assert_eq!(segment.end, Duration::from_millis(120));
        session.input.close().await.unwrap();
    }

    #[tokio::test]
    async fn timestamp_is_quantized_to_pcm_sample_boundary() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.2; 1],
                    start_sample: 0,
                    end_sample: 1,
                }]],
            }),
        )
        .unwrap();
        let mut session = transcriber.start();

        session
            .input
            .send_at(Duration::from_nanos(62_501), vec![0.2; 1])
            .await
            .unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = event else {
            panic!("expected quantized segment");
        };
        assert_eq!(segment.start, Duration::from_nanos(62_500));
    }

    #[tokio::test]
    async fn closing_one_source_flushes_it_while_primary_stays_open() {
        let model = FakeModel {
            calls: Arc::new(Mutex::new(Vec::new())),
            next: 0,
        };
        let system_source_vad = FinishVad {
            chunks: vec![vad::SpeechChunk {
                samples: vec![0.2; 8],
                start_sample: 0,
                end_sample: 8,
            }],
        };
        let transcriber = Transcriber {
            input_capacity: 32,
            max_sources: 2,
            model: Box::new(model),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![Box::new(system_source_vad)],
            }),
        };

        let mut session = transcriber.start();
        let system_input = session.open_source().await.unwrap();
        let system_source = system_input.source_id();
        system_input.send(vec![0.2; 8]).await.unwrap();
        system_input.close().await.unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        assert!(matches!(
            event,
            TranscriptEvent::Segment(TranscriptSegment { source_id, .. })
                if source_id == system_source
        ));

        session.input.send(vec![0.0; 4]).await.unwrap();
        session.input.close().await.unwrap();
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
    }

    #[tokio::test]
    async fn source_limit_is_enforced_and_released_on_close() {
        let mut config = test_config();
        config.max_sources = 2;
        let transcriber = Transcriber {
            input_capacity: config.input_capacity,
            max_sources: config.max_sources,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![
                    Box::new(FakeVad { chunks: Vec::new() }),
                    Box::new(FakeVad { chunks: Vec::new() }),
                ],
            }),
        };

        let mut session = transcriber.start();
        let first = session.open_source().await.unwrap();
        let first_id = first.source_id();
        let err = session.open_source().await.unwrap_err();
        assert!(matches!(err, Error::SourceLimit { max_sources: 2 }));

        first.close().await.unwrap();
        let replacement = session.open_source().await.unwrap();
        assert_ne!(replacement.source_id(), first_id);
        replacement.close().await.unwrap();
        session.input.close().await.unwrap();
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
    }

    #[tokio::test]
    async fn close_completes_before_events_are_drained() {
        let mut config = test_config();
        config.input_capacity = 1;
        let transcriber = Transcriber {
            input_capacity: config.input_capacity,
            max_sources: config.max_sources,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FinishVad {
                chunks: vec![
                    vad::SpeechChunk {
                        samples: vec![0.1; 4],
                        start_sample: 0,
                        end_sample: 4,
                    },
                    vad::SpeechChunk {
                        samples: vec![0.2; 4],
                        start_sample: 4,
                        end_sample: 8,
                    },
                ],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
        };
        let (input, mut events, worker) = transcriber.start().into_parts();
        let monitor = events.monitor();

        input.close().await.unwrap();
        worker.await.unwrap();
        let queued = monitor.metrics();
        assert_eq!(queued.pending_events, 3);
        assert_eq!(queued.peak_pending_events, 3);
        assert_eq!(queued.emitted_events, 3);
        assert_eq!(queued.received_events, 0);

        let first = events.recv().await.unwrap().unwrap();
        let second = events.recv().await.unwrap().unwrap();
        let end = events.recv().await.unwrap().unwrap();

        assert!(matches!(first, TranscriptEvent::Segment(_)));
        assert!(matches!(second, TranscriptEvent::Segment(_)));
        assert!(matches!(end, TranscriptEvent::EndOfStream));

        let drained = monitor.metrics();
        assert_eq!(drained.pending_events, 0);
        assert_eq!(drained.received_events, 3);
        assert_eq!(drained.discarded_events, 0);
        assert_eq!(drained.delivery_failures, 0);
    }

    #[tokio::test]
    async fn closing_event_receiver_cancels_idle_session() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad { chunks: Vec::new() }),
        )
        .unwrap();
        let mut session = transcriber.start();
        let monitor = session.events.monitor();

        session.events.close();
        assert!(matches!(
            session.open_source().await,
            Err(Error::StreamClosed)
        ));
        assert!(matches!(
            session.input.send(vec![0.0; 4]).await,
            Err(Error::StreamClosed)
        ));

        let (input, mut events, worker) = session.into_parts();
        assert_eq!(events.recv().await, None);
        assert!(matches!(input.close().await, Err(Error::StreamClosed)));
        worker.await.unwrap();

        let metrics = monitor.metrics();
        assert!(metrics.receiver_closed);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.emitted_events, 0);
    }

    #[tokio::test]
    async fn dropping_event_receiver_cancels_idle_session() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad { chunks: Vec::new() }),
        )
        .unwrap();
        let (input, events, worker) = transcriber.start().into_parts();
        let monitor = events.monitor();

        drop(events);
        for _ in 0..2_000 {
            if worker.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(worker.is_finished(), "worker did not observe receiver drop");
        worker.await.unwrap();

        assert!(matches!(
            input.send(vec![0.0; 4]).await,
            Err(Error::StreamClosed)
        ));
        assert!(matches!(input.close().await, Err(Error::StreamClosed)));

        let metrics = monitor.metrics();
        assert!(metrics.receiver_closed);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.emitted_events, 0);
        assert_eq!(metrics.discarded_events, 0);
        assert_eq!(metrics.delivery_failures, 0);
    }

    #[tokio::test]
    async fn dropping_receiver_cancels_inference_with_full_command_queue() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 1,
            model: Box::new(BlockingModel {
                started: started_tx,
                release: release_rx,
            }),
            vad: Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }]],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
        };
        let (input, events, worker) = transcriber.start().into_parts();
        let monitor = events.monitor();

        input.send(vec![0.1; 4]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        // Fill the command queue while inference holds the worker. The
        // receiver's best-effort Cancel command cannot be queued in this state,
        // so the shared cancellation flag must still stop the worker.
        input.send(vec![0.2; 4]).await.unwrap();
        drop(events);
        release_tx.send(()).unwrap();

        worker.await.unwrap();
        assert!(matches!(
            input.send(vec![0.3; 4]).await,
            Err(Error::StreamClosed)
        ));

        let metrics = monitor.metrics();
        assert!(metrics.receiver_closed);
        assert_eq!(metrics.emitted_events, 0);
        assert_eq!(metrics.pending_events, 0);
    }

    #[tokio::test]
    async fn dropping_partially_moved_last_input_closes_full_command_queue() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 1,
            model: Box::new(BlockingModel {
                started: started_tx,
                release: release_rx,
            }),
            vad: Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }]],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
        };
        let session = transcriber.start();
        let input = session.input;
        let mut events = session.events;
        let worker = session.worker;

        input.send(vec![0.1; 4]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        input.send(vec![0.2; 4]).await.unwrap();
        drop(input);
        release_tx.send(()).unwrap();

        for _ in 0..2_000 {
            if worker.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            worker.is_finished(),
            "worker did not observe the closed input"
        );
        worker.await.unwrap();

        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        assert_eq!(events.recv().await, None);
    }

    #[test]
    fn device_parse_accepts_documented_values() {
        assert_eq!("auto".parse::<Device>().unwrap(), Device::Auto);
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("directml".parse::<Device>().unwrap(), Device::DirectMl);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("tensorrt".parse::<Device>().unwrap(), Device::TensorRt);
        assert_eq!("openvino".parse::<Device>().unwrap(), Device::OpenVino);
        assert_eq!("rocm".parse::<Device>().unwrap(), Device::Rocm);
        assert_eq!("coreml".parse::<Device>().unwrap(), Device::CoreMl);
        assert_eq!("xnnpack".parse::<Device>().unwrap(), Device::Xnnpack);
        assert_eq!("onednn".parse::<Device>().unwrap(), Device::OneDnn);
        assert!("bogus".parse::<Device>().is_err());
    }
}
