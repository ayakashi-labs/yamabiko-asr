//! Parakeet-family on-device transcription for desktop applications.
//!
//! The crate owns the PCM streaming API, VAD gating, timestamp accounting, and
//! Parakeet TDT inference. Audio capture, resampling, downmix, and model
//! download stay with the application.

mod backend;
mod builder;
mod config;
mod error;
mod event;
mod session;
mod tdt;
mod vad;

pub use builder::{TranscriberBuilder, VadConfigBuilder};
pub use config::{
    AudioSourceConfig, AudioSourceId, AudioSourceKind, Device, Language, PCM_CHANNELS,
    PCM_SAMPLE_RATE_HZ, PcmChunk, PcmFormat, TranscriberConfig, VadConfig,
};
pub use error::{Error, Result};
pub use event::{
    SegmentId, SpeakerId, TranscriptEvent, TranscriptEventPayload, TranscriptSegment,
    TranscriptSegmentPayload,
};
pub use session::{AudioInput, TranscriptionSession, TranscriptionWorker};

use backend::{ParakeetTdtModel, RecognitionStream};
use session::{SessionCommand, run_transcription_worker};
use vad::{SileroVadFactory, SileroVadGate, VadFactory};

/// A transcription engine backed by one loaded Parakeet ASR model.
///
/// Multiple audio sources share this model while keeping independent VAD,
/// buffering, and source-local timeline state. Sources are scheduled
/// independently and are not mixed before transcription.
pub struct Transcriber {
    config: TranscriberConfig,
    model: Box<dyn backend::AsrModel>,
    stream: RecognitionStream,
    vad: Box<dyn vad::VadGate>,
    vad_factory: Box<dyn VadFactory>,
}

impl Transcriber {
    /// Load the ASR model and VAD backend from a validated configuration.
    ///
    /// Model loading is synchronous and may take long enough that GUI
    /// applications should call this away from their UI thread.
    pub fn new(config: TranscriberConfig) -> Result<Self> {
        config.validate()?;
        let model: Box<dyn backend::AsrModel> = Box::new(ParakeetTdtModel::load(&config)?);
        let vad = Box::new(SileroVadGate::new(config.vad.clone())?);
        let vad_factory = Box::new(SileroVadFactory::new(config.vad.clone()));
        Ok(Self {
            config,
            model,
            stream: RecognitionStream::default(),
            vad,
            vad_factory,
        })
    }

    /// Start the Tokio-facing streaming input/output API.
    ///
    /// The worker runs on Tokio's blocking pool because ONNX inference is
    /// synchronous. Input and output are bounded channels to provide natural
    /// backpressure for GUI applications.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime.
    pub fn start(self) -> TranscriptionSession {
        let (command_tx, command_rx) =
            tokio::sync::mpsc::channel::<SessionCommand>(self.config.channel_capacity);
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(self.config.channel_capacity);
        let worker = tokio::task::spawn_blocking(move || {
            run_transcription_worker(
                self.config,
                self.model,
                self.stream,
                self.vad,
                self.vad_factory,
                command_rx,
                event_tx,
            );
        });

        TranscriptionSession {
            input: AudioInput::new(AudioSourceId::PRIMARY, command_tx.clone()),
            events: event_rx,
            commands: command_tx,
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
            vad_factory: Box::new(SileroVadFactory::new(config.vad.clone())),
            config,
            model,
            stream: RecognitionStream::default(),
            vad,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct FakeModel {
        calls: Arc<Mutex<Vec<usize>>>,
        next: usize,
    }

    impl backend::AsrModel for FakeModel {
        fn transcribe(&mut self, samples: Vec<f32>, _language: &Language) -> Result<String> {
            self.calls.lock().unwrap().push(samples.len());
            self.next += 1;
            Ok(format!("chunk{}", self.next))
        }
    }

    struct FakeVad {
        chunks: Vec<Vec<vad::SpeechChunk>>,
    }

    impl vad::VadGate for FakeVad {
        fn push(&mut self, _chunk: &PcmChunk, _start_sample: u64) -> Result<Vec<vad::SpeechChunk>> {
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
        fn push(&mut self, _chunk: &PcmChunk, _start_sample: u64) -> Result<Vec<vad::SpeechChunk>> {
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
                    is_final: false,
                }],
            ],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session
            .input
            .send(PcmChunk::new(vec![0.0; 4]))
            .await
            .unwrap();
        session
            .input
            .send(PcmChunk::new(vec![0.2; 4]))
            .await
            .unwrap();
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
                is_final: true,
            }]],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session
            .input
            .send(PcmChunk::new(vec![0.2; 8]))
            .await
            .unwrap();
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
                is_final: true,
            }]],
        };
        let transcriber =
            Transcriber::from_parts(test_config(), Box::new(model), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session
            .input
            .send(PcmChunk::new(vec![0.1; 8]))
            .await
            .unwrap();
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
                is_final: true,
            }]],
        };
        let vad_factory = FakeVadFactory {
            vads: vec![Box::new(FakeVad {
                chunks: vec![vec![vad::SpeechChunk {
                    samples: vec![0.2; 8],
                    start_sample: 0,
                    end_sample: 8,
                    is_final: true,
                }]],
            })],
        };
        let transcriber = Transcriber {
            config: test_config(),
            model: Box::new(model),
            stream: RecognitionStream::default(),
            vad: Box::new(primary_vad),
            vad_factory: Box::new(vad_factory),
        };

        let mut session = transcriber.start();
        let system_input = session
            .open_source(AudioSourceConfig::system_audio())
            .await
            .unwrap();
        let system_source = system_input.source_id();
        session
            .input
            .send(PcmChunk::new(vec![0.1; 4]))
            .await
            .unwrap();
        system_input
            .send_at(Duration::from_millis(350), PcmChunk::new(vec![0.2; 8]))
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
    async fn discontinuous_explicit_timestamp_is_terminal_error() {
        let transcriber = Transcriber::from_parts(
            test_config(),
            Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            Box::new(FakeVad {
                chunks: vec![Vec::new(), Vec::new()],
            }),
        )
        .unwrap();
        let mut session = transcriber.start();

        session
            .input
            .send_at(Duration::from_millis(100), PcmChunk::new(vec![0.0; 160]))
            .await
            .unwrap();
        session
            .input
            .send_at(Duration::from_millis(105), PcmChunk::new(vec![0.0; 160]))
            .await
            .unwrap();

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
                        is_final: true,
                    }],
                ],
            }),
        )
        .unwrap();
        let mut session = transcriber.start();

        session
            .input
            .send_at(Duration::from_millis(100), PcmChunk::new(vec![0.0; 160]))
            .await
            .unwrap();
        session
            .input
            .send(PcmChunk::new(vec![0.2; 160]))
            .await
            .unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        let TranscriptEvent::Segment(segment) = event else {
            panic!("expected anchored segment");
        };
        assert_eq!(segment.start, Duration::from_millis(110));
        assert_eq!(segment.end, Duration::from_millis(120));
        session.input.close().await.unwrap();
    }

    #[tokio::test]
    async fn timestamp_must_align_to_pcm_sample_boundary() {
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

        session
            .input
            .send_at(Duration::from_nanos(1), PcmChunk::new(vec![0.0; 1]))
            .await
            .unwrap();

        let error = session.events.recv().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidTimestamp {
                source_id: AudioSourceId::PRIMARY,
                timestamp,
                ..
            } if timestamp == Duration::from_nanos(1)
        ));
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
                is_final: true,
            }],
        };
        let transcriber = Transcriber {
            config: test_config(),
            model: Box::new(model),
            stream: RecognitionStream::default(),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![Box::new(system_source_vad)],
            }),
        };

        let mut session = transcriber.start();
        let system_input = session
            .open_source(AudioSourceConfig::system_audio())
            .await
            .unwrap();
        let system_source = system_input.source_id();
        system_input
            .send(PcmChunk::new(vec![0.2; 8]))
            .await
            .unwrap();
        system_input.close().await.unwrap();

        let event = session.events.recv().await.unwrap().unwrap();
        assert!(matches!(
            event,
            TranscriptEvent::Segment(TranscriptSegment { source_id, .. })
                if source_id == system_source
        ));

        session
            .input
            .send(PcmChunk::new(vec![0.0; 4]))
            .await
            .unwrap();
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
            config,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            stream: RecognitionStream::default(),
            vad: Box::new(FakeVad { chunks: Vec::new() }),
            vad_factory: Box::new(FakeVadFactory {
                vads: vec![
                    Box::new(FakeVad { chunks: Vec::new() }),
                    Box::new(FakeVad { chunks: Vec::new() }),
                ],
            }),
        };

        let mut session = transcriber.start();
        let first = session
            .open_source(AudioSourceConfig::system_audio())
            .await
            .unwrap();
        let first_id = first.source_id();
        let err = session
            .open_source(AudioSourceConfig::other())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::SourceLimit { max_sources: 2 }));

        first.close().await.unwrap();
        let replacement = session
            .open_source(AudioSourceConfig::other())
            .await
            .unwrap();
        assert_ne!(replacement.source_id(), first_id);
        replacement.close().await.unwrap();
        session.input.close().await.unwrap();
        assert!(matches!(
            session.events.recv().await.unwrap().unwrap(),
            TranscriptEvent::EndOfStream
        ));
    }

    #[tokio::test]
    async fn close_completes_with_event_backpressure_when_events_are_drained() {
        let mut config = test_config();
        config.channel_capacity = 1;
        let transcriber = Transcriber {
            config,
            model: Box::new(FakeModel {
                calls: Arc::new(Mutex::new(Vec::new())),
                next: 0,
            }),
            stream: RecognitionStream::default(),
            vad: Box::new(FinishVad {
                chunks: vec![
                    vad::SpeechChunk {
                        samples: vec![0.1; 4],
                        start_sample: 0,
                        end_sample: 4,
                        is_final: true,
                    },
                    vad::SpeechChunk {
                        samples: vec![0.2; 4],
                        start_sample: 4,
                        end_sample: 8,
                        is_final: true,
                    },
                ],
            }),
            vad_factory: Box::new(FakeVadFactory { vads: Vec::new() }),
        };
        let (input, mut events, worker) = transcriber.start().into_parts();

        let producer = tokio::spawn(async move { input.close().await });
        let first = events.recv().await.unwrap().unwrap();
        let second = events.recv().await.unwrap().unwrap();
        let end = events.recv().await.unwrap().unwrap();

        assert!(matches!(first, TranscriptEvent::Segment(_)));
        assert!(matches!(second, TranscriptEvent::Segment(_)));
        assert!(matches!(end, TranscriptEvent::EndOfStream));
        producer.await.unwrap().unwrap();
        worker.await.unwrap();
    }

    #[test]
    fn config_validation_rejects_wrong_pcm_format() {
        let mut config = test_config();
        config.pcm_format = PcmFormat {
            sample_rate_hz: 48_000,
            channels: 1,
        };

        assert!(matches!(
            config.validate(),
            Err(Error::PcmFormat {
                expected: _,
                actual: _
            })
        ));
    }

    #[test]
    fn config_validation_rejects_bad_language_hint() {
        let mut config = test_config();
        config.language = Language::Hint("".to_string());

        assert!(matches!(
            config.validate(),
            Err(Error::InvalidLanguageHint(_))
        ));
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
