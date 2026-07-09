use crate::{Error, Result};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Required input sample rate for v0.1.
pub const PCM_SAMPLE_RATE_HZ: u32 = 16_000;

/// Required channel count for v0.1.
pub const PCM_CHANNELS: u16 = 1;

/// The only PCM format accepted by v0.1: f32, mono, 16 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl Default for PcmFormat {
    fn default() -> Self {
        Self {
            sample_rate_hz: PCM_SAMPLE_RATE_HZ,
            channels: PCM_CHANNELS,
        }
    }
}

impl PcmFormat {
    pub fn validate(self) -> Result<()> {
        let expected = Self::default();
        if self == expected {
            Ok(())
        } else {
            Err(Error::PcmFormat {
                expected,
                actual: self,
            })
        }
    }
}

impl fmt::Display for PcmFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f32 mono={} {}Hz", self.channels, self.sample_rate_hz)
    }
}

/// One chunk of f32 PCM on the input audio timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct PcmChunk {
    pub samples: Vec<f32>,
    pub format: PcmFormat,
}

impl PcmChunk {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            format: PcmFormat::default(),
        }
    }

    pub fn with_format(samples: Vec<f32>, format: PcmFormat) -> Self {
        Self { samples, format }
    }
}

/// Optional language target for multilingual models.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    Auto,
    Hint(String),
}

impl Language {
    pub fn hint(value: impl Into<String>) -> Result<Self> {
        let hint = value.into();
        Self::validate_hint(&hint)?;
        Ok(Self::Hint(normalize_language_hint(&hint)))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Auto => Ok(()),
            Self::Hint(hint) => Self::validate_hint(hint),
        }
    }

    fn validate_hint(hint: &str) -> Result<()> {
        let normalized = normalize_language_hint(hint);
        let valid = !normalized.is_empty()
            && normalized.len() <= 16
            && normalized
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-');

        if valid {
            Ok(())
        } else {
            Err(Error::InvalidLanguageHint(hint.to_string()))
        }
    }
}

/// Preferred execution device. Provider fallback follows the backend's
/// capabilities and should be surfaced by backend errors when unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    DirectMl,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::DirectMl => f.write_str("directml"),
        }
    }
}

/// Voice activity detection settings exposed by v0.1.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_speech: Duration,
    pub min_silence: Duration,
    pub speech_pad: Duration,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech: Duration::from_millis(250),
            min_silence: Duration::from_millis(100),
            speech_pad: Duration::from_millis(30),
        }
    }
}

impl VadConfig {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.threshold) || !self.threshold.is_finite() {
            return Err(Error::InvalidConfig(
                "VAD threshold must be a finite value from 0.0 to 1.0".to_string(),
            ));
        }
        if self.min_speech.is_zero() {
            return Err(Error::InvalidConfig(
                "VAD min_speech must be greater than zero".to_string(),
            ));
        }
        if self.min_silence.is_zero() {
            return Err(Error::InvalidConfig(
                "VAD min_silence must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

fn normalize_language_hint(hint: &str) -> String {
    match hint.trim() {
        "ja" => "ja-JP".to_string(),
        "en" => "en-US".to_string(),
        other => other.to_string(),
    }
}

/// Configuration for loading and running one Parakeet TDT transcriber.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriberConfig {
    pub model_dir: PathBuf,
    pub device: Device,
    pub language: Language,
    pub vad: VadConfig,
    pub pcm_format: PcmFormat,
    pub channel_capacity: usize,
}

impl TranscriberConfig {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            device: Device::default(),
            language: Language::default(),
            vad: VadConfig::default(),
            pcm_format: PcmFormat::default(),
            channel_capacity: 32,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_dir.as_os_str().is_empty() {
            return Err(Error::InvalidConfig(
                "model_dir must point to a local model directory".to_string(),
            ));
        }
        if self.channel_capacity == 0 {
            return Err(Error::InvalidConfig(
                "channel_capacity must be greater than zero".to_string(),
            ));
        }
        self.pcm_format.validate()?;
        self.language.validate()?;
        self.vad.validate()
    }
}
