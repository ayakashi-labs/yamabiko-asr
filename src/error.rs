use crate::{AudioSourceId, Device};
use std::fmt;
use std::time::Duration;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors exposed to applications using the Parakeet transcription pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A configuration value was outside the supported range.
    InvalidConfig(String),
    /// The requested execution device could not be used.
    DeviceUnavailable { device: Device, message: String },
    /// The ASR model could not be loaded.
    ModelLoad(String),
    /// The speaker diarization model could not be loaded.
    DiarizationModelLoad(String),
    /// VAD initialization or inference failed.
    Vad(String),
    /// ASR inference failed after model load.
    Backend(String),
    /// Speaker diarization inference failed after model load.
    Diarization(String),
    /// The session cannot accept another concurrently active source.
    SourceLimit { max_sources: usize },
    /// A command referenced a source that is no longer active.
    SourceNotFound { source_id: AudioSourceId },
    /// A session timestamp could not be represented on the 16 kHz timeline.
    InvalidTimestamp {
        source_id: AudioSourceId,
        timestamp: Duration,
        message: String,
    },
    /// An explicit chunk timestamp did not continue the source timeline.
    TimestampDiscontinuity {
        source_id: AudioSourceId,
        expected: Duration,
        actual: Duration,
    },
    /// The transcription worker is no longer accepting commands.
    StreamClosed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::DeviceUnavailable { device, message } => {
                write!(f, "execution device {device} is unavailable: {message}")
            }
            Self::ModelLoad(message) => write!(f, "failed to load ASR model: {message}"),
            Self::DiarizationModelLoad(message) => {
                write!(f, "failed to load diarization model: {message}")
            }
            Self::Vad(message) => write!(f, "VAD failed: {message}"),
            Self::Backend(message) => write!(f, "ASR backend failed: {message}"),
            Self::Diarization(message) => write!(f, "speaker diarization failed: {message}"),
            Self::SourceLimit { max_sources } => {
                write!(f, "audio source limit reached: max_sources={max_sources}")
            }
            Self::SourceNotFound { source_id } => {
                write!(f, "audio source {} is not active", source_id.get())
            }
            Self::InvalidTimestamp {
                source_id,
                timestamp,
                message,
            } => write!(
                f,
                "invalid timestamp {timestamp:?} for audio source {}: {message}",
                source_id.get()
            ),
            Self::TimestampDiscontinuity {
                source_id,
                expected,
                actual,
            } => write!(
                f,
                "timestamp discontinuity for audio source {}: expected {expected:?}, got {actual:?}",
                source_id.get()
            ),
            Self::StreamClosed => write!(f, "transcription stream closed"),
        }
    }
}

impl std::error::Error for Error {}
