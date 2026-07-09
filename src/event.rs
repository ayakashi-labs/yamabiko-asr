use std::time::Duration;

/// Events emitted by a running transcription session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptEvent {
    Segment(TranscriptSegment),
    EndOfStream,
}

/// One transcription segment on the input audio timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start: Duration,
    pub end: Duration,
    pub is_final: bool,
}
