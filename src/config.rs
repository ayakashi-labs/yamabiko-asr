use crate::{Error, Result};
use silero::{SampleRate, SpeechOptions};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

/// Required input sample rate.
pub const PCM_SAMPLE_RATE_HZ: u32 = 16_000;

/// Stable identifier for one audio source in a transcription session.
///
/// Source `0` is reserved for the primary input. Additional identifiers are
/// allocated by `TranscriptionSession::open_source`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct AudioSourceId(u64);

impl AudioSourceId {
    /// The primary input created with each transcription session.
    pub const PRIMARY: Self = Self(0);

    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the numeric representation used in UI payloads.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Preferred execution device for the Parakeet TDT ONNX sessions.
///
/// `Auto` tries available accelerated providers before CPU. Explicit
/// accelerator choices return a device error when their provider cannot be
/// registered. Silero VAD uses its bundled model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Device {
    /// Use the built-in ONNX Runtime CPU provider.
    #[default]
    Cpu,
    /// Try enabled Windows-capable providers in priority order, then use CPU.
    Auto,
    /// Use DirectML on a DirectX 12-capable GPU (`directml` feature).
    DirectMl,
    /// Use an NVIDIA CUDA GPU (`cuda` feature).
    Cuda,
    /// Use NVIDIA TensorRT with CUDA fallback (`tensorrt` feature).
    TensorRt,
    /// Use OpenVINO on supported Intel hardware (`openvino` feature).
    OpenVino,
    /// Use a Qualcomm NPU on Windows ARM64 (`qnn` feature).
    Qnn,
    /// Use an AMD Ryzen AI NPU (`vitis` feature).
    VitisAi,
    /// Use NVIDIA TensorRT RTX on a supported RTX GPU (`nvrtx` feature).
    NvRtx,
    /// Use native WebGPU through Dawn (`webgpu` feature).
    WebGpu,
    /// Use a TVM-enabled ONNX Runtime build (`tvm` feature).
    Tvm,
    /// Use the XNNPACK CPU provider (`xnnpack` feature).
    Xnnpack,
    /// Use the oneDNN CPU provider (`onednn` feature).
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
            Self::Qnn => f.write_str("qnn"),
            Self::VitisAi => f.write_str("vitis"),
            Self::NvRtx => f.write_str("nvrtx"),
            Self::WebGpu => f.write_str("webgpu"),
            Self::Tvm => f.write_str("tvm"),
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
            "qnn" => Ok(Self::Qnn),
            "vitis" | "vitisai" => Ok(Self::VitisAi),
            "nvrtx" => Ok(Self::NvRtx),
            "webgpu" | "wgpu" => Ok(Self::WebGpu),
            "tvm" => Ok(Self::Tvm),
            "xnnpack" => Ok(Self::Xnnpack),
            "onednn" | "dnnl" => Ok(Self::OneDnn),
            other => Err(Error::InvalidConfig(format!(
                "unsupported device '{other}'; expected one of auto, cpu, directml, cuda, tensorrt, openvino, qnn, vitis, nvrtx, webgpu, tvm, xnnpack, onednn"
            ))),
        }
    }
}

/// Voice activity detection settings.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VadConfig {
    pub(crate) threshold: f32,
    pub(crate) min_speech: Duration,
    pub(crate) min_silence: Duration,
    pub(crate) speech_pad: Duration,
    /// Maximum duration of one ASR utterance before it is split.
    pub(crate) max_speech: Duration,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech: Duration::from_millis(250),
            min_silence: Duration::from_millis(100),
            speech_pad: Duration::from_millis(30),
            max_speech: Duration::from_secs(30),
        }
    }
}

impl VadConfig {
    pub(crate) fn validate(&self) -> Result<SpeechOptions> {
        if !(0.0..=1.0).contains(&self.threshold) || !self.threshold.is_finite() {
            return Err(Error::InvalidConfig(
                "VAD threshold must be a finite value from 0.0 to 1.0".to_string(),
            ));
        }
        let options = self.options();
        if options.min_speech_samples() == 0 {
            return Err(Error::InvalidConfig(
                "VAD min_speech must be at least one millisecond".to_string(),
            ));
        }
        if options.min_silence_samples() == 0 {
            return Err(Error::InvalidConfig(
                "VAD min_silence must be at least one millisecond".to_string(),
            ));
        }
        if !self.speech_pad.is_zero() && options.speech_pad_samples() == 0 {
            return Err(Error::InvalidConfig(
                "VAD speech_pad must be zero or at least one millisecond".to_string(),
            ));
        }
        if self.max_speech.is_zero() {
            return Err(Error::InvalidConfig(
                "VAD max_speech must be greater than zero".to_string(),
            ));
        }
        if options
            .max_speech_samples()
            .is_some_and(|max| max < options.min_speech_samples())
        {
            return Err(Error::InvalidConfig(
                "VAD max_speech is too short for min_speech and speech_pad".to_string(),
            ));
        }
        Ok(options)
    }

    fn options(&self) -> SpeechOptions {
        SpeechOptions::new()
            .with_sample_rate(SampleRate::Rate16k)
            .with_start_threshold(self.threshold)
            .with_min_speech_duration(self.min_speech)
            .with_min_silence_duration(self.min_silence)
            .with_speech_pad(self.speech_pad)
            .with_max_speech_duration(self.max_speech)
    }
}

/// Configuration for loading and running one Parakeet TDT transcriber.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TranscriberConfig {
    pub(crate) model_dir: PathBuf,
    pub(crate) device: Device,
    pub(crate) vad: VadConfig,
    pub(crate) input_capacity: usize,
    pub(crate) max_sources: usize,
}

impl TranscriberConfig {
    pub(crate) fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            device: Device::default(),
            vad: VadConfig::default(),
            input_capacity: 32,
            max_sources: 2,
        }
    }

    pub(crate) fn validate(&self) -> Result<SpeechOptions> {
        if self.model_dir.as_os_str().is_empty() {
            return Err(Error::InvalidConfig(
                "model_dir must point to a local model directory".to_string(),
            ));
        }
        if self.input_capacity == 0 {
            return Err(Error::InvalidConfig(
                "input_capacity must be greater than zero".to_string(),
            ));
        }
        if self.max_sources == 0 {
            return Err(Error::InvalidConfig(
                "max_sources must be greater than zero".to_string(),
            ));
        }
        self.vad.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_zero_max_sources() {
        let mut config = TranscriberConfig::new("model");
        config.max_sources = 0;
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));
    }

    #[test]
    fn vad_rejects_max_speech_shorter_than_emittable_segment() {
        let mut config = VadConfig {
            max_speech: Duration::from_millis(341),
            ..VadConfig::default()
        };
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));

        config.max_speech = Duration::from_millis(342);
        config.validate().unwrap();
    }

    #[test]
    fn vad_rejects_nonzero_durations_truncated_to_zero_samples() {
        let sub_millisecond = Duration::from_nanos(999_999);

        for config in [
            VadConfig {
                min_speech: sub_millisecond,
                ..VadConfig::default()
            },
            VadConfig {
                min_silence: sub_millisecond,
                ..VadConfig::default()
            },
            VadConfig {
                speech_pad: sub_millisecond,
                ..VadConfig::default()
            },
        ] {
            assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));
        }

        let config = VadConfig {
            min_speech: Duration::from_millis(1),
            min_silence: Duration::from_millis(1),
            speech_pad: Duration::ZERO,
            ..VadConfig::default()
        };
        config.validate().unwrap();
    }
}
