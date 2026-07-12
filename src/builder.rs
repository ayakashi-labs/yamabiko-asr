use crate::{Device, Language, PcmFormat, Result, Transcriber, TranscriberConfig, VadConfig};
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

    pub fn language(mut self, language: Language) -> Self {
        self.config.language = language;
        self
    }

    pub fn language_hint(mut self, hint: impl Into<String>) -> Result<Self> {
        self.config.language = Language::hint(hint)?;
        Ok(self)
    }

    pub fn vad_config(mut self, vad: VadConfig) -> Self {
        self.config.vad = vad;
        self
    }

    pub fn vad(mut self, configure: impl FnOnce(VadConfigBuilder) -> VadConfigBuilder) -> Self {
        self.config.vad = configure(VadConfigBuilder::from(self.config.vad)).build();
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

    pub fn pcm_format(mut self, pcm_format: PcmFormat) -> Self {
        self.config.pcm_format = pcm_format;
        self
    }

    pub fn channel_capacity(mut self, channel_capacity: usize) -> Self {
        self.config.channel_capacity = channel_capacity;
        self
    }

    pub fn max_sources(mut self, max_sources: usize) -> Self {
        self.config.max_sources = max_sources;
        self
    }

    pub fn build_config(self) -> Result<TranscriberConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    pub fn build(self) -> Result<Transcriber> {
        Transcriber::new(self.config)
    }
}

/// Builder for VAD settings used inside `TranscriberBuilder::vad`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VadConfigBuilder {
    config: VadConfig,
}

impl From<VadConfig> for VadConfigBuilder {
    fn from(config: VadConfig) -> Self {
        Self { config }
    }
}

impl VadConfigBuilder {
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = threshold;
        self
    }

    pub fn min_speech(mut self, min_speech: Duration) -> Self {
        self.config.min_speech = min_speech;
        self
    }

    pub fn min_silence(mut self, min_silence: Duration) -> Self {
        self.config.min_silence = min_silence;
        self
    }

    pub fn speech_pad(mut self, speech_pad: Duration) -> Self {
        self.config.speech_pad = speech_pad;
        self
    }

    pub fn build(self) -> VadConfig {
        self.config
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
        let config = Transcriber::builder("model")
            .device(Device::Auto)
            .language_hint("ja")
            .unwrap()
            .vad(|vad| {
                vad.threshold(0.4)
                    .min_speech(Duration::from_millis(300))
                    .min_silence(Duration::from_millis(800))
                    .speech_pad(Duration::from_millis(40))
            })
            .channel_capacity(8)
            .max_sources(4)
            .build_config()
            .unwrap();

        assert_eq!(config.device, Device::Auto);
        assert_eq!(config.language, Language::Hint("ja-JP".to_string()));
        assert_eq!(config.vad.threshold, 0.4);
        assert_eq!(config.vad.min_speech, Duration::from_millis(300));
        assert_eq!(config.vad.min_silence, Duration::from_millis(800));
        assert_eq!(config.vad.speech_pad, Duration::from_millis(40));
        assert_eq!(config.channel_capacity, 8);
        assert_eq!(config.max_sources, 4);
    }

    #[test]
    fn transcriber_builder_validates_config_on_build_config() {
        let err = Transcriber::builder("model")
            .vad_threshold(f32::NAN)
            .build_config()
            .unwrap_err();

        assert!(matches!(err, Error::InvalidConfig(_)));
    }
}
