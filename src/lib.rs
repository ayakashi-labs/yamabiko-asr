//! Parakeet-family on-device transcription for desktop applications.
//!
//! The crate owns the PCM streaming API, VAD gating, timestamp accounting, and
//! Parakeet TDT inference. Audio capture, resampling, downmix, and model
//! download stay with the application.

mod backend;
mod config;
mod error;
mod event;
mod session;
mod tdt;
mod vad;

pub use config::{
    Device, Language, PCM_CHANNELS, PCM_SAMPLE_RATE_HZ, PcmChunk, PcmFormat, TranscriberConfig,
    VadConfig,
};
pub use error::{Error, Result};
pub use event::{TranscriptEvent, TranscriptSegment};
pub use session::TranscriptionSession;

use backend::ParakeetTdtBackend;
use session::run_transcription_worker;
use vad::SileroVadGate;

/// A single Parakeet transcription engine.
///
/// One `Transcriber` owns exactly one ASR backend instance. Future multi-PCM
/// support should merge or schedule input before it reaches this type; it
/// should not create hidden extra ASR engines.
pub struct Transcriber {
    config: TranscriberConfig,
    backend: Box<dyn backend::AsrBackend>,
    vad: Box<dyn vad::VadGate>,
}

impl Transcriber {
    /// Load the ASR model and VAD backend from a validated configuration.
    pub fn new(config: TranscriberConfig) -> Result<Self> {
        config.validate()?;
        let backend: Box<dyn backend::AsrBackend> = Box::new(ParakeetTdtBackend::load(&config)?);
        let vad = Box::new(SileroVadGate::new(config.vad.clone())?);
        Ok(Self {
            config,
            backend,
            vad,
        })
    }

    /// Start the Tokio-facing streaming input/output API.
    ///
    /// The worker runs on Tokio's blocking pool because ONNX inference is
    /// synchronous. Input and output are bounded channels to provide natural
    /// backpressure for GUI applications.
    pub fn start(self) -> TranscriptionSession {
        let (input_tx, input_rx) = tokio::sync::mpsc::channel(self.config.channel_capacity);
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(self.config.channel_capacity);
        let worker = tokio::task::spawn_blocking(move || {
            run_transcription_worker(self.config, self.backend, self.vad, input_rx, event_tx);
        });

        TranscriptionSession {
            input: input_tx,
            events: event_rx,
            worker,
        }
    }

    #[cfg(test)]
    fn from_parts(
        config: TranscriberConfig,
        backend: Box<dyn backend::AsrBackend>,
        vad: Box<dyn vad::VadGate>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            backend,
            vad,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct FakeBackend {
        calls: Arc<Mutex<Vec<usize>>>,
        next: usize,
    }

    impl backend::AsrBackend for FakeBackend {
        fn accept_speech(
            &mut self,
            speech: &vad::SpeechChunk,
            _language: &Language,
        ) -> Result<Vec<backend::BackendTranscript>> {
            self.calls.lock().unwrap().push(speech.samples.len());
            self.next += 1;
            Ok(vec![backend::BackendTranscript {
                text: format!("chunk{}", self.next),
                start_sample: speech.start_sample,
                end_sample: speech.end_sample,
                is_final: speech.is_final,
            }])
        }

        fn flush(&mut self, _next_input_sample: u64) -> Result<Vec<backend::BackendTranscript>> {
            Ok(Vec::new())
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

    fn test_config() -> TranscriberConfig {
        TranscriberConfig::new("model-dir")
    }

    #[tokio::test]
    async fn vad_gating_keeps_silence_out_of_backend() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let backend = FakeBackend {
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
            Transcriber::from_parts(test_config(), Box::new(backend), Box::new(vad)).unwrap();

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
        drop(session.input);

        while let Some(event) = session.events.recv().await {
            if matches!(event.unwrap(), TranscriptEvent::EndOfStream) {
                break;
            }
        }

        assert_eq!(*calls.lock().unwrap(), vec![4]);
    }

    #[tokio::test]
    async fn timestamps_use_input_audio_timeline() {
        let backend = FakeBackend {
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
            Transcriber::from_parts(test_config(), Box::new(backend), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session
            .input
            .send(PcmChunk::new(vec![0.2; 8]))
            .await
            .unwrap();
        drop(session.input);

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
        let backend = FakeBackend {
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
            Transcriber::from_parts(test_config(), Box::new(backend), Box::new(vad)).unwrap();

        let mut session = transcriber.start();
        session
            .input
            .send(PcmChunk::new(vec![0.1; 8]))
            .await
            .unwrap();
        drop(session.input);

        let first = session.events.recv().await.unwrap().unwrap();
        let second = session.events.recv().await.unwrap().unwrap();

        assert!(matches!(
            first,
            TranscriptEvent::Segment(TranscriptSegment { is_final: true, .. })
        ));
        assert!(matches!(second, TranscriptEvent::EndOfStream));
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
