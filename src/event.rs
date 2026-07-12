use crate::AudioSourceId;
use std::time::Duration;

/// Stable identifier for a transcript segment within one session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentId(u64);

impl SegmentId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the numeric representation used in UI payloads.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable identifier for an anonymous or identified speaker.
///
/// Speaker assignment is optional until a diarization or identification
/// stage associates a segment with a speaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpeakerId(u64);

impl SpeakerId {
    /// Create an application- or diarizer-assigned speaker identifier.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the numeric representation used in UI payloads.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Events emitted by a running transcription session.
///
/// Consumers should treat `Segment` as an upsert keyed by `SegmentId` so a
/// future diarization or streaming decoder can revise text, timing, or speaker
/// assignment without introducing a second update mechanism.
#[non_exhaustive]
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

/// One transcription segment on its source-local audio timeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptSegment {
    /// Stable key for inserting or updating this segment.
    pub id: SegmentId,
    /// Audio source that produced this segment.
    pub source_id: AudioSourceId,
    /// Assigned speaker, when speaker processing is available.
    pub speaker_id: Option<SpeakerId>,
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
    pub id: u64,
    pub source_id: u64,
    pub speaker_id: Option<u64>,
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
            id: segment.id.get(),
            source_id: segment.source_id.get(),
            speaker_id: segment.speaker_id.map(SpeakerId::get),
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
            id: segment.id.get(),
            source_id: segment.source_id.get(),
            speaker_id: segment.speaker_id.map(SpeakerId::get),
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
            id: SegmentId::new(42),
            source_id: AudioSourceId::PRIMARY,
            speaker_id: Some(SpeakerId::new(3)),
            text: "hello".to_string(),
            start: Duration::from_millis(1_234),
            end: Duration::from_millis(2_500),
            inference_duration: Duration::from_millis(140),
            is_final: true,
        };

        let payload = segment.to_payload();

        assert_eq!(payload.id, 42);
        assert_eq!(payload.source_id, AudioSourceId::PRIMARY.get());
        assert_eq!(payload.speaker_id, Some(3));
        assert_eq!(payload.text, "hello");
        assert_eq!(payload.start_ms, 1_234);
        assert_eq!(payload.end_ms, 2_500);
        assert_eq!(payload.duration_ms, 1_266);
        assert_eq!(payload.inference_ms, 140);
        assert!(payload.is_final);
    }
}
