use crate::{Error, PCM_SAMPLE_RATE_HZ, Result, VadConfig};
use silero::{SampleRate, Session, SpeechSegment, SpeechSegmenter, StreamState};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeechChunk {
    pub samples: Vec<f32>,
    pub start_sample: u64,
    pub end_sample: u64,
}

pub(crate) trait VadGate: Send {
    fn push(&mut self, samples: &[f32], start_sample: u64) -> Result<Vec<SpeechChunk>>;
    fn finish(&mut self) -> Result<Vec<SpeechChunk>>;
}

pub(crate) trait VadFactory: Send {
    fn create(&mut self) -> Result<Box<dyn VadGate>>;
}

pub(crate) struct SileroVadFactory {
    config: VadConfig,
    session: Arc<Mutex<Session>>,
}

impl SileroVadFactory {
    pub(crate) fn new(config: VadConfig) -> Result<Self> {
        config.validate()?;
        let session = Session::bundled().map_err(|err| Error::Vad(err.to_string()))?;
        Ok(Self {
            config,
            session: Arc::new(Mutex::new(session)),
        })
    }
}

impl VadFactory for SileroVadFactory {
    fn create(&mut self) -> Result<Box<dyn VadGate>> {
        Ok(Box::new(SileroVadGate::new(
            self.config.clone(),
            Arc::clone(&self.session),
        )?))
    }
}

pub(crate) struct SileroVadGate {
    session: Arc<Mutex<Session>>,
    stream: StreamState,
    segmenter: SpeechSegmenter,
    buffer: PcmBuffer,
    next_emit_sample: u64,
}

impl SileroVadGate {
    fn new(config: VadConfig, session: Arc<Mutex<Session>>) -> Result<Self> {
        let options = config.speech_options()?;
        let stream = StreamState::new(SampleRate::Rate16k);
        let segmenter = SpeechSegmenter::new(options);

        Ok(Self {
            session,
            stream,
            segmenter,
            buffer: PcmBuffer::default(),
            next_emit_sample: 0,
        })
    }

    fn push_segment(&mut self, segment: SpeechSegment, out: &mut Vec<SpeechChunk>) -> Result<()> {
        let segment_start = segment.start_sample();
        // Silero pads the final partial frame with zeroes, so its reported end
        // can extend past the PCM actually supplied by the caller.
        let end = segment.end_sample().min(self.buffer.end_sample());
        let payload_start = segment_start.max(self.next_emit_sample);
        if end <= payload_start {
            return Ok(());
        }

        let samples = self.buffer.take(payload_start, end).ok_or_else(|| {
            Error::Vad(format!(
                "VAD segment {payload_start}..{end} is outside the retained PCM buffer"
            ))
        })?;

        self.next_emit_sample = end;
        out.push(SpeechChunk {
            samples,
            start_sample: payload_start,
            end_sample: end,
        });
        Ok(())
    }

    fn detect_segments(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Vad("shared VAD session lock poisoned".to_string()))?;
        let mut segments = Vec::new();
        let mut maybe_segment = self
            .segmenter
            .push_samples(&mut session, &mut self.stream, samples)
            .map_err(|err| Error::Vad(err.to_string()))?;

        while let Some(segment) = maybe_segment {
            segments.push(segment);
            maybe_segment = self
                .segmenter
                .push_samples(&mut session, &mut self.stream, &[])
                .map_err(|err| Error::Vad(err.to_string()))?;
        }
        Ok(segments)
    }

    fn finish_segments(&mut self) -> Result<Vec<SpeechSegment>> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Vad("shared VAD session lock poisoned".to_string()))?;
        let mut segments = Vec::new();
        let mut maybe_segment = self
            .segmenter
            .finish_stream(&mut session, &mut self.stream)
            .map_err(|err| Error::Vad(err.to_string()))?;

        while let Some(segment) = maybe_segment {
            segments.push(segment);
            maybe_segment = self
                .segmenter
                .push_samples(&mut session, &mut self.stream, &[])
                .map_err(|err| Error::Vad(err.to_string()))?;
        }
        Ok(segments)
    }
}

impl VadGate for SileroVadGate {
    fn push(&mut self, samples: &[f32], start_sample: u64) -> Result<Vec<SpeechChunk>> {
        self.buffer.append(start_sample, samples);

        let mut out = Vec::new();
        for segment in self.detect_segments(samples)? {
            self.push_segment(segment, &mut out)?;
        }

        if !self.segmenter.is_active() {
            let input_end = start_sample.saturating_add(samples.len() as u64);
            let processed_until = input_end.saturating_sub(self.stream.pending_len() as u64);
            let keep_from =
                processed_until.saturating_sub(self.segmenter.options().speech_pad_samples());
            self.buffer.discard_before(keep_from);
        }

        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<SpeechChunk>> {
        let mut out = Vec::new();
        for segment in self.finish_segments()? {
            self.push_segment(segment, &mut out)?;
        }

        self.buffer.clear();
        Ok(out)
    }
}

#[derive(Default)]
struct PcmBuffer {
    samples: VecDeque<f32>,
    start_sample: u64,
}

impl PcmBuffer {
    fn append(&mut self, start_sample: u64, samples: &[f32]) {
        if self.samples.is_empty() {
            self.start_sample = start_sample;
        }
        self.samples.extend(samples.iter().copied());
    }

    fn take(&mut self, start: u64, end: u64) -> Option<Vec<f32>> {
        let rel_start = usize::try_from(start.checked_sub(self.start_sample)?).ok()?;
        let rel_end = usize::try_from(end.checked_sub(self.start_sample)?).ok()?;
        if rel_end > self.samples.len() || rel_start >= rel_end {
            return None;
        }

        self.samples.drain(..rel_start);
        let samples = self.samples.drain(..rel_end - rel_start).collect();
        self.start_sample = end;
        Some(samples)
    }

    fn end_sample(&self) -> u64 {
        self.start_sample.saturating_add(self.samples.len() as u64)
    }

    fn discard_before(&mut self, sample: u64) {
        let end = self.end_sample();
        let target = sample.clamp(self.start_sample, end);
        let count = (target - self.start_sample) as usize;
        self.samples.drain(..count);
        self.start_sample = target;
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

pub(crate) fn duration_from_samples(samples: u64) -> std::time::Duration {
    let nanos = samples as u128 * 1_000_000_000u128 / PCM_SAMPLE_RATE_HZ as u128;
    std::time::Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

#[cfg(test)]
mod tests {
    use super::{PcmBuffer, SileroVadFactory, SileroVadGate, VadGate};
    use crate::VadConfig;
    use std::sync::Arc;

    #[test]
    fn pcm_buffer_discards_silence_without_shifting_payload_timestamps() {
        let mut buffer = PcmBuffer::default();
        buffer.append(0, &[0.0; 1_000]);
        buffer.discard_before(900);
        buffer.append(1_000, &[1.0; 100]);

        assert_eq!(buffer.samples.len(), 200);
        assert_eq!(buffer.take(950, 1_050).unwrap().len(), 100);
        assert_eq!(buffer.start_sample, 1_050);
    }

    #[test]
    fn source_gates_share_session_and_bound_silence_buffer() {
        let config = VadConfig::default();
        let factory = SileroVadFactory::new(config.clone()).unwrap();
        let mut gate = SileroVadGate::new(config.clone(), Arc::clone(&factory.session)).unwrap();
        let second = SileroVadGate::new(config, Arc::clone(&factory.session)).unwrap();
        assert!(Arc::ptr_eq(&gate.session, &second.session));
        drop(second);

        let chunk = vec![0.0; 1_600];

        for index in 0..100 {
            gate.push(&chunk, index * chunk.len() as u64).unwrap();
        }

        assert!(!gate.segmenter.is_active());
        let retained_limit =
            gate.segmenter.options().speech_pad_samples() as usize + gate.stream.pending_len();
        assert!(gate.buffer.samples.len() <= retained_limit);
    }

    #[test]
    fn final_segment_is_clamped_to_supplied_pcm() {
        let config = VadConfig::default();
        let factory = SileroVadFactory::new(config.clone()).unwrap();
        let mut gate = SileroVadGate::new(config, Arc::clone(&factory.session)).unwrap();
        gate.buffer.append(0, &[1.0; 100]);

        let mut chunks = Vec::new();
        gate.push_segment(
            silero::SpeechSegment::new(0, 512, silero::SampleRate::Rate16k),
            &mut chunks,
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), 100);
        assert_eq!(chunks[0].end_sample, 100);
    }
}
