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

mod audio_features;
mod backend;
mod builder;
mod config;
mod diarization;
mod error;
mod event;
mod ort_utils;
mod session;
mod sortformer;
mod tdt;
mod vad;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

pub use builder::TranscriberBuilder;
pub(crate) use config::TranscriberConfig;
pub use config::{AudioSourceId, Device, DiarizationConfig, PCM_SAMPLE_RATE_HZ};
pub use diarization::{AudioSourceOptions, DiarizationMode};
pub use error::{Error, Result};
pub use event::{SegmentId, SpeakerActivity, SpeakerId, TranscriptEvent, TranscriptSegment};
pub use session::{
    AudioInput, OutputMetrics, OutputMonitor, TranscriptEventReceiver, TranscriptionSession,
    TranscriptionWorker,
};

use diarization::{DiarizerFactory, UnavailableDiarizerFactory};
use session::{
    SessionCommand, TranscriptionWorkerParams, command_channel, output_channel,
    run_diarization_worker, run_transcription_worker,
};
use sortformer::SortformerDiarizerFactory;
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
    diarizer_factory: Box<dyn DiarizerFactory>,
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
        let diarizer_factory: Box<dyn DiarizerFactory> = match config.diarization {
            Some(diarization) => {
                let device = diarization.effective_device(config.device);
                Box::new(SortformerDiarizerFactory::new(
                    diarization.model_dir,
                    device,
                    config.max_sources,
                )?)
            }
            None => Box::new(UnavailableDiarizerFactory),
        };
        Ok(Self {
            input_capacity: config.input_capacity,
            max_sources: config.max_sources,
            model,
            vad,
            vad_factory: Box::new(vad_factory),
            diarizer_factory,
        })
    }

    /// Start the Tokio-facing streaming input/output API.
    ///
    /// The worker runs on Tokio's blocking pool because ONNX inference is
    /// synchronous. Audio commands have bounded capacity to provide natural
    /// backpressure; control and transcript event delivery remain unbounded so
    /// closing an input never depends on queue capacity or draining output.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime.
    pub fn start(self) -> TranscriptionSession {
        self.start_with_options(AudioSourceOptions::OFF)
            .expect("the default non-diarized source is initialized during construction")
    }

    /// Start a session with options for its primary audio source.
    ///
    /// The call is fallible because enabling diarization loads the configured
    /// model lazily. A load failure rejects this session without changing the
    /// behavior of [`Self::start`], whose primary source remains non-diarized.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime.
    pub fn start_with_options(
        self,
        primary_options: AudioSourceOptions,
    ) -> Result<TranscriptionSession> {
        let wait_for_diarizer = matches!(
            primary_options.diarization,
            diarization::DiarizationMode::On
        );
        let (command_tx, command_rx) = command_channel(self.input_capacity);
        let (event_tx, events, cancelled) = output_channel(command_tx.downgrade());
        let (diarization_tx, diarization_rx) = tokio::sync::mpsc::unbounded_channel();
        let job_capacity = std::sync::Arc::new(tokio::sync::Semaphore::new(self.input_capacity));
        let runtime = tokio::runtime::Handle::current();
        let internal_tx = command_tx.internal_sender();
        let panic_tx = internal_tx.clone();
        let diarizer_factory = self.diarizer_factory;
        let diarization_cancelled = std::sync::Arc::clone(&cancelled);
        let diarization_worker = std::thread::Builder::new()
            .name("yamabiko-diarization".to_string())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_diarization_worker(
                        diarizer_factory,
                        diarization_rx,
                        internal_tx,
                        job_capacity,
                        runtime,
                        diarization_cancelled,
                    );
                }));
                if result.is_err() {
                    let _ = panic_tx.send(SessionCommand::DiarizationFailed(Error::Diarization(
                        "diarization worker panicked".to_string(),
                    )));
                }
                result
            })
            .expect("failed to spawn diarization worker");

        let (startup_tx, startup_rx) = std::sync::mpsc::channel();
        let (transcription_done_tx, transcription_done_rx) = std::sync::mpsc::channel();
        let worker_cancelled = std::sync::Arc::clone(&cancelled);
        tokio::task::spawn_blocking(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_transcription_worker(TranscriptionWorkerParams {
                    max_sources: self.max_sources,
                    model: self.model,
                    primary_vad: self.vad,
                    vad_factory: self.vad_factory,
                    primary_options,
                    diarization_tx,
                    command_rx,
                    event_tx,
                    cancelled: worker_cancelled,
                    startup_reply: startup_tx,
                });
            }));
            let _ = transcription_done_tx.send(result);
        });
        let worker = tokio::task::spawn_blocking(move || {
            let transcription_result = transcription_done_rx
                .recv()
                .expect("transcription worker completion channel closed");
            let diarization_result = match diarization_worker.join() {
                Ok(result) => result,
                Err(payload) => Err(payload),
            };
            if let Err(payload) = transcription_result {
                std::panic::resume_unwind(payload);
            }
            if let Err(payload) = diarization_result {
                std::panic::resume_unwind(payload);
            }
        });

        if wait_for_diarizer {
            startup_rx.recv().map_err(|_| Error::StreamClosed)??;
        }

        Ok(TranscriptionSession {
            input: AudioInput::new(AudioSourceId::PRIMARY, command_tx, cancelled),
            events,
            worker,
        })
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
    use diarization::fake::{
        Behavior as FakeDiarizerBehavior, factory as fake_diarizer_factory,
        factory_with_capacity as fake_diarizer_factory_with_capacity,
        failing_factory as failing_diarizer_factory, retrying_factory as retrying_diarizer_factory,
    };
    use std::sync::atomic::Ordering;
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
    async fn all_off_sources_do_not_initialize_diarization() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (diarizer_factory, creates, opened, pushes) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(1)],
            });
        let transcriber = Transcriber {
            input_capacity: 4,
            max_sources: 2,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }]],
            }),
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![Box::new(FakeVad {
                    chunks: vec![vec![vad::SpeechChunk {
                        samples: vec![0.2; 6],
                        start_sample: 0,
                        end_sample: 6,
                    }]],
                })],
            }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber.start();
        let additional = session.open_source().await.unwrap();
        session.input.send(vec![0.1; 4]).await.unwrap();
        additional.send(vec![0.2; 6]).await.unwrap();
        additional.close().await.unwrap();
        session.input.close().await.unwrap();

        let mut speakers = Vec::new();
        while let Some(event) = session.events.recv().await {
            match event.unwrap() {
                TranscriptEvent::SpeakerActivity(_) => {
                    panic!("OFF sources must not emit speaker activity")
                }
                TranscriptEvent::Segment(segment) => speakers.push(segment.speaker_id),
                TranscriptEvent::EndOfStream => break,
            }
        }
        session.worker.await.unwrap();

        assert_eq!(creates.load(Ordering::SeqCst), 0);
        assert!(opened.lock().unwrap().is_empty());
        assert!(pushes.lock().unwrap().is_empty());
        assert_eq!(*calls.lock().unwrap(), vec![4, 6]);
        assert_eq!(speakers, vec![None, None]);
    }

    #[tokio::test]
    async fn enabling_diarization_without_builder_config_rejects_the_source() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad { chunks: Vec::new() }),
        )
        .unwrap();

        let error = transcriber
            .start_with_options(AudioSourceOptions::new().diarization(DiarizationMode::On))
            .err()
            .unwrap();

        assert_eq!(
            error,
            Error::InvalidConfig(
                "speaker diarization is not configured on this transcriber".to_string()
            )
        );
    }

    #[tokio::test]
    async fn mixed_sources_route_only_enabled_pcm_through_diarization() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (diarizer_factory, creates, opened, pushes) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(7)],
            });
        let transcriber = Transcriber {
            input_capacity: 4,
            max_sources: 2,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }]],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber.start();
        let additional = session
            .open_source_with_options(AudioSourceOptions::ON)
            .await
            .unwrap();
        let additional_id = additional.source_id();
        session.input.send(vec![0.1; 4]).await.unwrap();
        additional.send(vec![0.2; 8]).await.unwrap();
        additional.close().await.unwrap();
        session.input.close().await.unwrap();

        let mut activities = Vec::new();
        let mut segments = Vec::new();
        while let Some(event) = session.events.recv().await {
            match event.unwrap() {
                TranscriptEvent::SpeakerActivity(activity) => activities.push(activity),
                TranscriptEvent::Segment(segment) => segments.push(segment),
                TranscriptEvent::EndOfStream => break,
            }
        }
        session.worker.await.unwrap();

        assert_eq!(creates.load(Ordering::SeqCst), 1);
        assert_eq!(*opened.lock().unwrap(), vec![additional_id]);
        assert_eq!(*pushes.lock().unwrap(), vec![(additional_id, vec![0.2; 8])]);
        assert_eq!(*calls.lock().unwrap(), vec![4, 8]);
        assert_eq!(segments.len(), 2);
        let primary = segments
            .iter()
            .find(|segment| segment.source_id == AudioSourceId::PRIMARY)
            .unwrap();
        let diarized = segments
            .iter()
            .find(|segment| segment.source_id == additional_id)
            .unwrap();
        assert_eq!(primary.speaker_id, None);
        assert_eq!(diarized.speaker_id, Some(SpeakerId::new(0)));
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].source_id, additional_id);
        assert_eq!(activities[0].speaker_id, SpeakerId::new(0));
    }

    #[tokio::test]
    async fn enabled_sources_share_one_worker_and_use_session_unique_speaker_ids() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (diarizer_factory, creates, opened, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(3)],
            });
        let transcriber = Transcriber {
            input_capacity: 4,
            max_sources: 2,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap();
        let additional = session
            .open_source_with_options(AudioSourceOptions::ON)
            .await
            .unwrap();
        let additional_id = additional.source_id();
        session.input.send(vec![0.1; 4]).await.unwrap();
        additional.send(vec![0.2; 4]).await.unwrap();
        additional.close().await.unwrap();
        session.input.close().await.unwrap();

        let mut activity_speaker_ids = Vec::new();
        let mut speaker_ids = Vec::new();
        while let Some(event) = session.events.recv().await {
            match event.unwrap() {
                TranscriptEvent::SpeakerActivity(activity) => {
                    activity_speaker_ids.push(activity.speaker_id)
                }
                TranscriptEvent::Segment(segment) => speaker_ids.push(segment.speaker_id.unwrap()),
                TranscriptEvent::EndOfStream => break,
            }
        }
        session.worker.await.unwrap();

        assert_eq!(creates.load(Ordering::SeqCst), 1);
        assert_eq!(
            *opened.lock().unwrap(),
            vec![AudioSourceId::PRIMARY, additional_id]
        );
        assert_eq!(*calls.lock().unwrap(), vec![4, 4]);
        speaker_ids.sort_by_key(|id| id.get());
        assert_eq!(speaker_ids, vec![SpeakerId::new(0), SpeakerId::new(1)]);
        activity_speaker_ids.sort_by_key(|id| id.get());
        assert_eq!(activity_speaker_ids, speaker_ids);
    }

    #[tokio::test]
    async fn overlapping_speech_creates_one_unattributed_asr_job() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (diarizer_factory, _, _, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![
                    diarization::BackendSpeakerId::new(1),
                    diarization::BackendSpeakerId::new(2),
                ],
            });
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap();
        session.input.send(vec![0.4; 5]).await.unwrap();
        session.input.close().await.unwrap();
        let first = session.events.recv().await.unwrap().unwrap();
        let second = session.events.recv().await.unwrap().unwrap();
        let activity_speakers = [first, second].map(|event| match event {
            TranscriptEvent::SpeakerActivity(activity) => activity.speaker_id,
            _ => panic!("expected overlapping speaker activity before the transcript"),
        });
        assert_eq!(activity_speakers, [SpeakerId::new(0), SpeakerId::new(1)]);
        let segment = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = segment else {
            panic!("expected one segment")
        };
        assert_eq!(segment.speaker_id, None);
        assert_eq!(*calls.lock().unwrap(), vec![5]);
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        session.worker.await.unwrap();
    }

    #[tokio::test]
    async fn speaker_activity_is_available_while_asr_is_still_running() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (diarizer_factory, _, _, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(2)],
            });
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 1,
            model: Box::new(BlockingModel {
                started: started_tx,
                release: release_rx,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();

        input.send(vec![0.3; 8]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let activity = futures_util::FutureExt::now_or_never(events.recv())
            .expect("speaker activity must not wait for ASR")
            .unwrap()
            .unwrap();
        assert!(matches!(activity, TranscriptEvent::SpeakerActivity(_)));

        release_tx.send(()).unwrap();
        input.close().await.unwrap();
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn diarizer_initialization_failure_rejects_only_enabled_source() {
        let (diarizer_factory, creates) =
            failing_diarizer_factory(Error::Backend("fake initialization failure".to_string()));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 2,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.1; 4],
                    start_sample: 0,
                    end_sample: 4,
                }]],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber.start();
        assert_eq!(
            session
                .open_source_with_options(AudioSourceOptions::ON)
                .await
                .unwrap_err(),
            Error::Backend("fake initialization failure".to_string())
        );
        session.input.send(vec![0.1; 4]).await.unwrap();
        session.input.close().await.unwrap();
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        session.worker.await.unwrap();
        assert_eq!(creates.load(Ordering::SeqCst), 1);
        assert_eq!(*calls.lock().unwrap(), vec![4]);
    }

    #[tokio::test]
    async fn failed_diarizer_initialization_can_be_retried() {
        let (diarizer_factory, creates, opened, _) = retrying_diarizer_factory(
            Error::DiarizationModelLoad("temporary model load failure".to_string()),
            FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(4)],
            },
        );
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 2,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let mut session = transcriber.start();
        let options = AudioSourceOptions::new().diarization(DiarizationMode::On);

        assert!(matches!(
            session.open_source_with_options(options).await,
            Err(Error::DiarizationModelLoad(_))
        ));
        let source = session.open_source_with_options(options).await.unwrap();
        let source_id = source.source_id();
        source.send(vec![0.2; 4]).await.unwrap();
        source.close().await.unwrap();
        session.input.close().await.unwrap();

        assert_eq!(creates.load(Ordering::SeqCst), 2);
        assert_eq!(*opened.lock().unwrap(), vec![source_id]);
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::SpeakerActivity(_)
        ));
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        session.worker.await.unwrap();
    }

    #[tokio::test]
    async fn diarizer_runtime_failure_is_one_terminal_session_error() {
        let (diarizer_factory, _, _, _) = fake_diarizer_factory(FakeDiarizerBehavior::FailPush(
            Error::Backend("fake diarization runtime failure".to_string()),
        ));
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();
        input.send(vec![0.1; 4]).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap().unwrap_err(),
            Error::Backend("fake diarization runtime failure".to_string())
        );
        assert_eq!(events.recv().await, None);
        worker.await.unwrap();
        assert_eq!(input.close().await.unwrap_err(), Error::StreamClosed);
    }

    #[tokio::test]
    async fn enabled_source_close_flushes_final_job_before_reply() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (diarizer_factory, _, _, _) = fake_diarizer_factory(FakeDiarizerBehavior::Flush {
            speakers: vec![diarization::BackendSpeakerId::new(9)],
        });
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };

        let mut session = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap();
        session
            .input
            .send_at(Duration::from_millis(100), vec![0.1; 4])
            .await
            .unwrap();
        session.input.close().await.unwrap();
        assert_eq!(*calls.lock().unwrap(), vec![4]);

        let TranscriptEvent::SpeakerActivity(activity) =
            session.events.recv().await.unwrap().unwrap()
        else {
            panic!("expected speaker activity before the flushed segment")
        };
        assert_eq!(activity.speaker_id, SpeakerId::new(0));
        assert_eq!(activity.start, Duration::from_millis(100));
        assert_eq!(
            activity.end,
            Duration::from_millis(100) + Duration::from_micros(250)
        );
        let TranscriptEvent::Segment(segment) = session.events.recv().await.unwrap().unwrap()
        else {
            panic!("expected flushed segment")
        };
        assert_eq!(segment.speaker_id, Some(SpeakerId::new(0)));
        assert_eq!(segment.start, Duration::from_millis(100));
        assert_eq!(
            segment.end,
            Duration::from_millis(100) + Duration::from_micros(250)
        );
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        session.worker.await.unwrap();
    }

    #[tokio::test]
    async fn enabled_pcm_releases_input_capacity_when_worker_consumes_it() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (diarizer_factory, _, _, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::BlockThenFinalize {
                speakers: vec![diarization::BackendSpeakerId::new(1)],
                started: started_tx,
                release: release_rx,
            });
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();

        input.send(vec![0.1; 4]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        {
            let second_send = input.send(vec![0.2; 4]);
            tokio::pin!(second_send);
            assert_eq!(
                futures_util::FutureExt::now_or_never(second_send.as_mut()),
                Some(Ok(()))
            );
        }
        release_tx.send(()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        release_tx.send(()).unwrap();
        input.close().await.unwrap();

        let mut segment_count = 0;
        while let Some(event) = events.recv().await {
            match event.unwrap() {
                TranscriptEvent::SpeakerActivity(_) => {}
                TranscriptEvent::Segment(_) => segment_count += 1,
                TranscriptEvent::EndOfStream => break,
            }
        }
        worker.await.unwrap();
        assert_eq!(segment_count, 2);
    }

    #[tokio::test]
    async fn diarizer_lookahead_can_exceed_input_command_capacity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (processed_tx, processed_rx) = std_mpsc::channel();
        let (diarizer_factory, _, _, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeAfter {
                pushes: 2,
                speakers: vec![diarization::BackendSpeakerId::new(1)],
                processed: processed_tx,
            });
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::clone(&calls),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();

        input.send(vec![0.1; 4]).await.unwrap();
        processed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        {
            let second_send = input.send(vec![0.2; 4]);
            tokio::pin!(second_send);
            assert_eq!(
                futures_util::FutureExt::now_or_never(second_send.as_mut()),
                Some(Ok(())),
                "lookahead input should not deadlock behind retained PCM"
            );
        }
        input.close().await.unwrap();

        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::SpeakerActivity(_)
        ));
        let TranscriptEvent::Segment(segment) = events.recv().await.unwrap().unwrap() else {
            panic!("expected one diarized segment")
        };
        assert_eq!(segment.speaker_id, Some(SpeakerId::new(0)));
        assert_eq!(*calls.lock().unwrap(), vec![8]);
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn diarizer_retained_pcm_is_bounded_separately_from_input_commands() {
        let (diarizer_factory, _, _, _) = fake_diarizer_factory_with_capacity(
            FakeDiarizerBehavior::Flush {
                speakers: vec![diarization::BackendSpeakerId::new(1)],
            },
            4,
        );
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();

        input.send(vec![0.1; 5]).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap().unwrap_err(),
            Error::Backend(
                "invalid diarization output: backend retained 5 PCM samples, exceeding its declared limit of 4"
                    .to_string()
            )
        );
        assert_eq!(events.recv().await, None);
        worker.await.unwrap();
        assert_eq!(input.close().await.unwrap_err(), Error::StreamClosed);
    }

    #[tokio::test]
    async fn enabled_source_reports_timestamp_discontinuity_as_terminal() {
        let (diarizer_factory, _, _, _) =
            fake_diarizer_factory(FakeDiarizerBehavior::FinalizeEach {
                speakers: vec![diarization::BackendSpeakerId::new(1)],
            });
        let transcriber = Transcriber {
            input_capacity: 2,
            max_sources: 1,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
            diarizer_factory: Box::new(diarizer_factory),
        };
        let (input, mut events, worker) = transcriber
            .start_with_options(AudioSourceOptions::ON)
            .unwrap()
            .into_parts();

        input
            .send_at(Duration::from_millis(10), vec![0.1; 4])
            .await
            .unwrap();
        input
            .send_at(Duration::from_millis(20), vec![0.2; 4])
            .await
            .unwrap();
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::SpeakerActivity(_)
        ));
        assert!(matches!(
            events.recv().await.unwrap().unwrap(),
            TranscriptEvent::Segment(_)
        ));
        assert!(matches!(
            events.recv().await.unwrap().unwrap_err(),
            Error::TimestampDiscontinuity {
                source_id: AudioSourceId::PRIMARY,
                ..
            }
        ));
        assert_eq!(events.recv().await, None);
        worker.await.unwrap();
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
    async fn dropping_receiver_cancels_inference_with_queued_audio() {
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
        };
        let (input, events, worker) = transcriber.start().into_parts();
        let monitor = events.monitor();

        input.send(vec![0.1; 4]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        // Queue more audio while inference holds the worker. Cancellation must
        // stop the worker before it processes that pending audio.
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
    async fn dropping_additional_input_closes_it_with_full_audio_queue() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let transcriber = Transcriber {
            input_capacity: 1,
            max_sources: 2,
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
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![
                    Box::new(FinishVad {
                        chunks: vec![vad::SpeechChunk {
                            samples: vec![0.2; 4],
                            start_sample: 0,
                            end_sample: 4,
                        }],
                    }),
                    Box::new(FakeVad { chunks: Vec::new() }),
                ],
            }),
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
        };
        let mut session = transcriber.start();
        let additional = session.open_source().await.unwrap();
        let additional_id = additional.source_id();

        session.input.send(vec![0.1; 4]).await.unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        // Fill the audio queue while inference holds the worker, then drop the
        // additional input. Its ordered close command must not be lost.
        additional.send(vec![0.2; 4]).await.unwrap();
        drop(additional);
        release_tx.send(()).unwrap();

        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        release_tx.send(()).unwrap();

        let first = session.events.recv().await.unwrap().unwrap();
        let second = session.events.recv().await.unwrap().unwrap();
        assert!(matches!(
            (first, second),
            (
                TranscriptEvent::Segment(TranscriptSegment {
                    source_id: AudioSourceId::PRIMARY,
                    ..
                }),
                TranscriptEvent::Segment(TranscriptSegment { source_id, .. })
            ) if source_id == additional_id
        ));

        let replacement = session.open_source().await.unwrap();
        assert_ne!(replacement.source_id(), additional_id);
        replacement.close().await.unwrap();
        session.input.close().await.unwrap();
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
    }

    #[tokio::test]
    async fn dropping_partially_moved_last_input_closes_with_full_audio_queue() {
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
            diarizer_factory: Box::new(UnavailableDiarizerFactory),
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
        for (name, device) in [
            ("auto", Device::Auto),
            ("cpu", Device::Cpu),
            ("directml", Device::DirectMl),
            ("cuda", Device::Cuda),
            ("tensorrt", Device::TensorRt),
            ("openvino", Device::OpenVino),
            ("qnn", Device::Qnn),
            ("vitis", Device::VitisAi),
            ("nvrtx", Device::NvRtx),
            ("webgpu", Device::WebGpu),
            ("tvm", Device::Tvm),
            ("xnnpack", Device::Xnnpack),
            ("onednn", Device::OneDnn),
        ] {
            assert_eq!(name.parse::<Device>().unwrap(), device);
            assert_eq!(device.to_string(), name);
        }

        for unsupported in ["rocm", "coreml", "azure", "bogus"] {
            assert!(unsupported.parse::<Device>().is_err());
        }
    }
}
