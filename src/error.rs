use crate::{Device, PcmFormat};
use std::fmt;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors exposed to applications using the streaming transcription pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Input PCM did not match the required v0.1 format.
    PcmFormat {
        expected: PcmFormat,
        actual: PcmFormat,
    },
    /// A language hint was empty or not accepted by the selected backend.
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
    /// The input or output stream was closed before processing completed.
    StreamClosed,
    /// The blocking transcription worker failed to join.
    Join(String),
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
            Self::StreamClosed => write!(f, "transcription stream closed"),
            Self::Join(message) => write!(f, "transcription worker failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}
