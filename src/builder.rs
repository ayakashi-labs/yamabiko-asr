use crate::{Device, Result, Transcriber, TranscriberConfig};
use std::path::Path;
use std::time::Duration;

/// Builder for creating a configured `Transcriber`.
#[derive(Debug, Clone)]
pub struct TranscriberBuilder {
    config: TranscriberConfig,
}

impl TranscriberBuilder {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            config: TranscriberConfig::new(model_dir),
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.config.vad.threshold = threshold;
        self
    }

    pub fn vad_min_speech(mut self, min_speech: Duration) -> Self {
        self.config.vad.min_speech = min_speech;
        self
    }

    pub fn vad_min_silence(mut self, min_silence: Duration) -> Self {
        self.config.vad.min_silence = min_silence;
        self
    }

    pub fn vad_speech_pad(mut self, speech_pad: Duration) -> Self {
        self.config.vad.speech_pad = speech_pad;
        self
    }

    pub fn vad_max_speech(mut self, max_speech: Duration) -> Self {
        self.config.vad.max_speech = max_speech;
        self
    }

    pub fn input_capacity(mut self, input_capacity: usize) -> Self {
        self.config.input_capacity = input_capacity;
        self
    }

    pub fn max_sources(mut self, max_sources: usize) -> Self {
        self.config.max_sources = max_sources;
        self
    }

    pub fn build(self) -> Result<Transcriber> {
        Transcriber::new(self.config)
    }
}

impl Transcriber {
    /// Start configuring a `Transcriber` from a model directory.
    pub fn builder(model_dir: impl AsRef<Path>) -> TranscriberBuilder {
        TranscriberBuilder::new(model_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn transcriber_builder_builds_validated_config() {
        let builder = Transcriber::builder("model")
            .device(Device::Auto)
            .vad_threshold(0.4)
            .vad_min_speech(Duration::from_millis(300))
            .vad_min_silence(Duration::from_millis(800))
            .vad_speech_pad(Duration::from_millis(40))
            .vad_max_speech(Duration::from_secs(20))
            .input_capacity(8)
            .max_sources(4);
        builder.config.validate().unwrap();
        let config = builder.config;

        assert_eq!(config.device, Device::Auto);
        assert_eq!(config.vad.threshold, 0.4);
        assert_eq!(config.vad.min_speech, Duration::from_millis(300));
        assert_eq!(config.vad.min_silence, Duration::from_millis(800));
        assert_eq!(config.vad.speech_pad, Duration::from_millis(40));
        assert_eq!(config.vad.max_speech, Duration::from_secs(20));
        assert_eq!(config.input_capacity, 8);
        assert_eq!(config.max_sources, 4);
    }

    #[test]
    fn transcriber_builder_validates_before_model_load() {
        let err = Transcriber::builder("model")
            .vad_threshold(f32::NAN)
            .build()
            .err()
            .unwrap();

        assert!(matches!(err, Error::InvalidConfig(_)));
    }
}
