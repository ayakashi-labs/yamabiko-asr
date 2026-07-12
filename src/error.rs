use crate::{AudioSourceId, Device, PcmFormat};
use std::fmt;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors exposed to applications using the Parakeet transcription pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Input PCM did not match the required v0.1 format.
    PcmFormat {
        expected: PcmFormat,
        actual: PcmFormat,
    },
    /// A language hint was empty or not accepted by the Parakeet backend.
    InvalidLanguageHint(String),
    /// A configuration value was outside the supported range.
    InvalidConfig(String),
    /// The requested execution device could not be used.
    DeviceUnavailable { device: Device, message: String },
    /// The ASR model could not be loaded.
    ModelLoad(String),
    /// VAD initialization or inference failed.
    Vad(String),
    /// ASR inference failed after model load.
    Backend(String),
    /// The session cannot accept another concurrently active source.
    SourceLimit { max_sources: usize },
    /// A command referenced a source that is no longer active.
    SourceNotFound { source_id: AudioSourceId },
    /// The transcription worker is no longer accepting commands.
    StreamClosed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PcmFormat { expected, actual } => write!(
                f,
                "unsupported PCM format: expected {expected}, got {actual}"
            ),
            Self::InvalidLanguageHint(hint) => write!(f, "invalid language hint: {hint}"),
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::DeviceUnavailable { device, message } => {
                write!(f, "execution device {device} is unavailable: {message}")
            }
            Self::ModelLoad(message) => write!(f, "failed to load ASR model: {message}"),
            Self::Vad(message) => write!(f, "VAD failed: {message}"),
            Self::Backend(message) => write!(f, "ASR backend failed: {message}"),
            Self::SourceLimit { max_sources } => {
                write!(f, "audio source limit reached: max_sources={max_sources}")
            }
            Self::SourceNotFound { source_id } => {
                write!(f, "audio source {} is not active", source_id.get())
            }
            Self::StreamClosed => write!(f, "transcription stream closed"),
        }
    }
}

impl std::error::Error for Error {}
