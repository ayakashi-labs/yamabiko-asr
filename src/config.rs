use crate::{Error, Result};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
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

/// Stable identifier for one audio source in a transcription session.
///
/// Source `0` is reserved for the primary input. Additional identifiers are
/// allocated by `TranscriptionSession::open_source`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioSourceId(u64);

impl AudioSourceId {
    /// The default source used by `PcmChunk::new`.
    pub const PRIMARY: Self = Self(0);

    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the numeric representation used in UI payloads.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Application-level role of an audio source.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AudioSourceKind {
    Microphone,
    SystemAudio,
    #[default]
    Other,
}

/// Configuration used when registering an additional audio source.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioSourceConfig {
    pub kind: AudioSourceKind,
}

impl AudioSourceConfig {
    pub fn microphone() -> Self {
        Self {
            kind: AudioSourceKind::Microphone,
        }
    }

    pub fn system_audio() -> Self {
        Self {
            kind: AudioSourceKind::SystemAudio,
        }
    }

    pub fn other() -> Self {
        Self::default()
    }
}

/// One chunk of f32 PCM on its source-local audio timeline.
///
/// Chunks for each source must be sent in capture order. Every source has an
/// independent sample counter starting at zero.
#[non_exhaustive]
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
#[non_exhaustive]
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

/// Preferred execution device for the Parakeet TDT ONNX sessions.
///
/// `Auto` tries available accelerated providers before CPU. Explicit
/// accelerator choices return a device error when their provider cannot be
/// registered. Silero VAD uses its bundled default session in v0.1.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    Auto,
    DirectMl,
    Cuda,
    TensorRt,
    OpenVino,
    Rocm,
    CoreMl,
    Xnnpack,
    OneDnn,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Auto => f.write_str("auto"),
            Self::DirectMl => f.write_str("directml"),
            Self::Cuda => f.write_str("cuda"),
            Self::TensorRt => f.write_str("tensorrt"),
            Self::OpenVino => f.write_str("openvino"),
            Self::Rocm => f.write_str("rocm"),
            Self::CoreMl => f.write_str("coreml"),
            Self::Xnnpack => f.write_str("xnnpack"),
            Self::OneDnn => f.write_str("onednn"),
        }
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "directml" | "dml" => Ok(Self::DirectMl),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" | "trt" => Ok(Self::TensorRt),
            "openvino" | "ov" => Ok(Self::OpenVino),
            "rocm" => Ok(Self::Rocm),
            "coreml" => Ok(Self::CoreMl),
            "xnnpack" => Ok(Self::Xnnpack),
            "onednn" | "dnnl" => Ok(Self::OneDnn),
            other => Err(Error::InvalidConfig(format!(
                "unsupported device '{other}'; expected one of auto, cpu, directml, cuda, tensorrt, openvino, rocm, coreml, xnnpack, onednn"
            ))),
        }
    }
}

/// Voice activity detection settings exposed by v0.1.
#[non_exhaustive]
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
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_min_speech(mut self, min_speech: Duration) -> Self {
        self.min_speech = min_speech;
        self
    }

    pub fn with_min_silence(mut self, min_silence: Duration) -> Self {
        self.min_silence = min_silence;
        self
    }

    pub fn with_speech_pad(mut self, speech_pad: Duration) -> Self {
        self.speech_pad = speech_pad;
        self
    }

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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriberConfig {
    pub model_dir: PathBuf,
    pub device: Device,
    pub language: Language,
    pub vad: VadConfig,
    pub pcm_format: PcmFormat,
    pub channel_capacity: usize,
    pub max_sources: usize,
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
            max_sources: 2,
        }
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_language(mut self, language: Language) -> Self {
        self.language = language;
        self
    }

    pub fn with_language_hint(mut self, hint: impl Into<String>) -> Result<Self> {
        self.language = Language::hint(hint)?;
        Ok(self)
    }

    pub fn with_vad(mut self, vad: VadConfig) -> Self {
        self.vad = vad;
        self
    }

    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad.threshold = threshold;
        self
    }

    pub fn with_vad_min_speech(mut self, min_speech: Duration) -> Self {
        self.vad.min_speech = min_speech;
        self
    }

    pub fn with_vad_min_silence(mut self, min_silence: Duration) -> Self {
        self.vad.min_silence = min_silence;
        self
    }

    pub fn with_vad_speech_pad(mut self, speech_pad: Duration) -> Self {
        self.vad.speech_pad = speech_pad;
        self
    }

    pub fn with_pcm_format(mut self, pcm_format: PcmFormat) -> Self {
        self.pcm_format = pcm_format;
        self
    }

    pub fn with_channel_capacity(mut self, channel_capacity: usize) -> Self {
        self.channel_capacity = channel_capacity;
        self
    }

    pub fn with_max_sources(mut self, max_sources: usize) -> Self {
        self.max_sources = max_sources;
        self
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
        if self.max_sources == 0 {
            return Err(Error::InvalidConfig(
                "max_sources must be greater than zero".to_string(),
            ));
        }
        self.pcm_format.validate()?;
        self.language.validate()?;
        self.vad.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_methods_set_values() {
        let vad = VadConfig::default()
            .with_threshold(0.4)
            .with_min_speech(Duration::from_millis(300))
            .with_min_silence(Duration::from_millis(800))
            .with_speech_pad(Duration::from_millis(40));

        let config = TranscriberConfig::new("model")
            .with_device(Device::Auto)
            .with_language(Language::Auto)
            .with_vad(vad.clone())
            .with_channel_capacity(8)
            .with_max_sources(4);

        assert_eq!(config.device, Device::Auto);
        assert_eq!(config.language, Language::Auto);
        assert_eq!(config.vad, vad);
        assert_eq!(config.channel_capacity, 8);
        assert_eq!(config.max_sources, 4);
    }

    #[test]
    fn config_builder_accepts_language_hint() {
        let config = TranscriberConfig::new("model")
            .with_language_hint("ja")
            .unwrap();

        assert_eq!(config.language, Language::Hint("ja-JP".to_string()));
    }

    #[test]
    fn config_rejects_zero_max_sources() {
        let config = TranscriberConfig::new("model").with_max_sources(0);
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));
    }
}
