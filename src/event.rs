use crate::AudioSourceId;
use std::time::Duration;

/// Stable identifier for a transcript segment within one session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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
/// Non-diarized and overlapping transcript segments intentionally have no
/// speaker assignment. Diarized single-speaker segments use the same ID as
/// their preceding [`SpeakerActivity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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
/// `SpeakerActivity` is finalized and precedes the transcript produced from
/// the same audio interval.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "type", content = "data", rename_all = "snake_case")
)]
pub enum TranscriptEvent {
    /// Finalized speaker activity emitted before its corresponding transcript.
    SpeakerActivity(SpeakerActivity),
    Segment(TranscriptSegment),
    EndOfStream,
}

/// One finalized speaker-active interval on the shared session timeline.
///
/// Adjacent intervals for the same source and speaker may be joined by the
/// consumer. Overlapping speech is represented by one activity event per
/// active speaker while the corresponding transcript remains unattributed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SpeakerActivity {
    /// Audio source on which this speaker was active.
    pub source_id: AudioSourceId,
    /// Session-unique speaker identifier.
    pub speaker_id: SpeakerId,
    /// Activity start timestamp on the session timeline.
    #[cfg_attr(
        feature = "serde",
        serde(rename = "start_ms", serialize_with = "serialize_duration_ms")
    )]
    pub start: Duration,
    /// Activity end timestamp on the session timeline.
    #[cfg_attr(
        feature = "serde",
        serde(rename = "end_ms", serialize_with = "serialize_duration_ms")
    )]
    pub end: Duration,
}

impl SpeakerActivity {
    /// Activity start timestamp in milliseconds on the session timeline.
    pub fn start_ms(&self) -> u64 {
        duration_ms(self.start)
    }

    /// Activity end timestamp in milliseconds on the session timeline.
    pub fn end_ms(&self) -> u64 {
        duration_ms(self.end)
    }

    /// Activity duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        duration_ms(self.end.saturating_sub(self.start))
    }
}

/// One transcription segment on the shared session audio timeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TranscriptSegment {
    /// Stable key for inserting or updating this segment.
    pub id: SegmentId,
    /// Audio source that produced this segment.
    pub source_id: AudioSourceId,
    /// Assigned speaker, when speaker processing is available.
    pub speaker_id: Option<SpeakerId>,
    pub text: String,
    #[cfg_attr(
        feature = "serde",
        serde(rename = "start_ms", serialize_with = "serialize_duration_ms")
    )]
    pub start: Duration,
    #[cfg_attr(
        feature = "serde",
        serde(rename = "end_ms", serialize_with = "serialize_duration_ms")
    )]
    pub end: Duration,
    /// Wall-clock time spent running ASR inference for this segment.
    #[cfg_attr(
        feature = "serde",
        serde(rename = "inference_ms", serialize_with = "serialize_duration_ms")
    )]
    pub inference_duration: Duration,
    pub is_final: bool,
}

impl TranscriptSegment {
    /// Segment start timestamp in milliseconds on the session timeline.
    pub fn start_ms(&self) -> u64 {
        duration_ms(self.start)
    }

    /// Segment end timestamp in milliseconds on the session timeline.
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
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(feature = "serde")]
fn serialize_duration_ms<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(duration_ms(*duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_helpers_use_millisecond_timestamps() {
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

        assert_eq!(segment.id.get(), 42);
        assert_eq!(segment.source_id, AudioSourceId::PRIMARY);
        assert_eq!(segment.speaker_id.map(SpeakerId::get), Some(3));
        assert_eq!(segment.start_ms(), 1_234);
        assert_eq!(segment.end_ms(), 2_500);
        assert_eq!(segment.duration_ms(), 1_266);
        assert_eq!(segment.inference_ms(), 140);
    }

    #[test]
    fn speaker_activity_helpers_use_millisecond_timestamps() {
        let activity = SpeakerActivity {
            source_id: AudioSourceId::PRIMARY,
            speaker_id: SpeakerId::new(7),
            start: Duration::from_millis(400),
            end: Duration::from_millis(1_250),
        };

        assert_eq!(activity.start_ms(), 400);
        assert_eq!(activity.end_ms(), 1_250);
        assert_eq!(activity.duration_ms(), 850);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn speaker_activity_serializes_like_existing_events() {
        let event = TranscriptEvent::SpeakerActivity(SpeakerActivity {
            source_id: AudioSourceId::PRIMARY,
            speaker_id: SpeakerId::new(7),
            start: Duration::from_millis(400),
            end: Duration::from_millis(1_250),
        });

        assert_eq!(
            serde_json::to_value(event).unwrap(),
            serde_json::json!({
                "type": "speaker_activity",
                "data": {
                    "source_id": 0,
                    "speaker_id": 7,
                    "start_ms": 400,
                    "end_ms": 1250
                }
            })
        );
    }
}
