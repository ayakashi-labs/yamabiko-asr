use std::time::Duration;

/// Events emitted by a running transcription session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptEvent {
    Segment(TranscriptSegment),
    EndOfStream,
}

impl TranscriptEvent {
    /// Convert this event into a UI-friendly payload.
    ///
    /// When the `serde` Cargo feature is enabled, the payload types implement
    /// `Serialize`/`Deserialize` and can be emitted directly from Tauri.
    pub fn to_payload(&self) -> TranscriptEventPayload {
        self.into()
    }
}

/// One transcription segment on the input audio timeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start: Duration,
    pub end: Duration,
    /// Wall-clock time spent running ASR inference for this segment.
    pub inference_duration: Duration,
    pub is_final: bool,
}

impl TranscriptSegment {
    /// Segment start timestamp in milliseconds on the input audio timeline.
    pub fn start_ms(&self) -> u64 {
        duration_ms(self.start)
    }

    /// Segment end timestamp in milliseconds on the input audio timeline.
    pub fn end_ms(&self) -> u64 {
        duration_ms(self.end)
    }

    /// Segment duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        duration_ms(self.end.saturating_sub(self.start))
    }

    /// ASR inference duration in milliseconds.
    pub fn inference_ms(&self) -> u64 {
        duration_ms(self.inference_duration)
    }

    /// Convert this segment into a UI-friendly payload.
    ///
    /// When the `serde` Cargo feature is enabled, the payload implements
    /// `Serialize`/`Deserialize` and can be emitted directly from Tauri.
    pub fn to_payload(&self) -> TranscriptSegmentPayload {
        self.into()
    }
}

/// UI-friendly transcription event payload.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "type", content = "data", rename_all = "snake_case")
)]
pub enum TranscriptEventPayload {
    Segment(TranscriptSegmentPayload),
    EndOfStream,
}

impl From<&TranscriptEvent> for TranscriptEventPayload {
    fn from(event: &TranscriptEvent) -> Self {
        match event {
            TranscriptEvent::Segment(segment) => Self::Segment(segment.into()),
            TranscriptEvent::EndOfStream => Self::EndOfStream,
        }
    }
}

impl From<TranscriptEvent> for TranscriptEventPayload {
    fn from(event: TranscriptEvent) -> Self {
        match event {
            TranscriptEvent::Segment(segment) => Self::Segment(segment.into()),
            TranscriptEvent::EndOfStream => Self::EndOfStream,
        }
    }
}

/// UI-friendly transcription segment payload using millisecond timestamps.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptSegmentPayload {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
    pub inference_ms: u64,
    pub is_final: bool,
}

impl From<&TranscriptSegment> for TranscriptSegmentPayload {
    fn from(segment: &TranscriptSegment) -> Self {
        Self {
            text: segment.text.clone(),
            start_ms: segment.start_ms(),
            end_ms: segment.end_ms(),
            duration_ms: segment.duration_ms(),
            inference_ms: segment.inference_ms(),
            is_final: segment.is_final,
        }
    }
}

impl From<TranscriptSegment> for TranscriptSegmentPayload {
    fn from(segment: TranscriptSegment) -> Self {
        Self {
            start_ms: segment.start_ms(),
            end_ms: segment.end_ms(),
            duration_ms: segment.duration_ms(),
            inference_ms: segment.inference_ms(),
            is_final: segment.is_final,
            text: segment.text,
        }
    }
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_payload_uses_millisecond_timestamps() {
        let segment = TranscriptSegment {
            text: "hello".to_string(),
            start: Duration::from_millis(1_234),
            end: Duration::from_millis(2_500),
            inference_duration: Duration::from_millis(140),
            is_final: true,
        };

        let payload = segment.to_payload();

        assert_eq!(payload.text, "hello");
        assert_eq!(payload.start_ms, 1_234);
        assert_eq!(payload.end_ms, 2_500);
        assert_eq!(payload.duration_ms, 1_266);
        assert_eq!(payload.inference_ms, 140);
        assert!(payload.is_final);
    }
}
