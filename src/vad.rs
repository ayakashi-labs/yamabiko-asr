use crate::{Error, PCM_SAMPLE_RATE_HZ, PcmChunk, Result, VadConfig};
use silero::{SampleRate, Session, SpeechOptions, SpeechSegment, SpeechSegmenter, StreamState};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeechChunk {
    pub samples: Vec<f32>,
    pub start_sample: u64,
    pub end_sample: u64,
    pub is_final: bool,
}

pub(crate) trait VadGate: Send {
    fn push(&mut self, chunk: &PcmChunk, start_sample: u64) -> Result<Vec<SpeechChunk>>;
    fn finish(&mut self) -> Result<Vec<SpeechChunk>>;
}

pub(crate) struct SileroVadGate {
    session: Session,
    stream: StreamState,
    segmenter: SpeechSegmenter,
    buffer: Vec<f32>,
    buffer_start_sample: u64,
    next_emit_sample: u64,
}

impl SileroVadGate {
    pub(crate) fn new(config: VadConfig) -> Result<Self> {
        config.validate()?;
        let options = SpeechOptions::new()
            .with_sample_rate(SampleRate::Rate16k)
            .with_start_threshold(config.threshold)
            .with_min_speech_duration(config.min_speech)
            .with_min_silence_duration(config.min_silence)
            .with_speech_pad(config.speech_pad);

        let session = Session::bundled().map_err(|err| Error::Vad(err.to_string()))?;
        let stream = StreamState::new(SampleRate::Rate16k);
        let segmenter = SpeechSegmenter::new(options);

        Ok(Self {
            session,
            stream,
            segmenter,
            buffer: Vec::new(),
            buffer_start_sample: 0,
            next_emit_sample: 0,
        })
    }

    fn push_segment(&mut self, segment: SpeechSegment, out: &mut Vec<SpeechChunk>) {
        let segment_start = segment.start_sample();
        let end = segment.end_sample();
        let payload_start = segment_start.max(self.next_emit_sample);
        if end <= payload_start {
            out.push(SpeechChunk {
                samples: Vec::new(),
                start_sample: segment_start,
                end_sample: end,
                is_final: true,
            });
            return;
        }

        let Some(samples) = self.slice_samples(payload_start, end) else {
            return;
        };

        self.next_emit_sample = end;
        self.drop_before(end);
        out.push(SpeechChunk {
            samples,
            start_sample: payload_start,
            end_sample: end,
            is_final: true,
        });
    }

    fn slice_samples(&self, start: u64, end: u64) -> Option<Vec<f32>> {
        let rel_start = start.checked_sub(self.buffer_start_sample)? as usize;
        let rel_end = end.checked_sub(self.buffer_start_sample)? as usize;
        if rel_end > self.buffer.len() || rel_start >= rel_end {
            return None;
        }
        Some(self.buffer[rel_start..rel_end].to_vec())
    }

    fn drop_before(&mut self, sample: u64) {
        if sample <= self.buffer_start_sample {
            return;
        }
        let drop = (sample - self.buffer_start_sample) as usize;
        if drop >= self.buffer.len() {
            self.buffer.clear();
            self.buffer_start_sample = sample;
        } else {
            self.buffer.drain(0..drop);
            self.buffer_start_sample = sample;
        }
    }
}

impl VadGate for SileroVadGate {
    fn push(&mut self, chunk: &PcmChunk, start_sample: u64) -> Result<Vec<SpeechChunk>> {
        if self.buffer.is_empty() {
            self.buffer_start_sample = start_sample;
        }
        self.buffer.extend_from_slice(&chunk.samples);

        let mut out = Vec::new();
        let mut maybe_segment = self
            .segmenter
            .push_samples(&mut self.session, &mut self.stream, &chunk.samples)
            .map_err(|err| Error::Vad(err.to_string()))?;

        while let Some(segment) = maybe_segment {
            self.push_segment(segment, &mut out);
            maybe_segment = self
                .segmenter
                .push_samples(&mut self.session, &mut self.stream, &[])
                .map_err(|err| Error::Vad(err.to_string()))?;
        }

        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<SpeechChunk>> {
        let mut out = Vec::new();
        let mut maybe_segment = self
            .segmenter
            .finish_stream(&mut self.session, &mut self.stream)
            .map_err(|err| Error::Vad(err.to_string()))?;

        while let Some(segment) = maybe_segment {
            self.push_segment(segment, &mut out);
            maybe_segment = self
                .segmenter
                .push_samples(&mut self.session, &mut self.stream, &[])
                .map_err(|err| Error::Vad(err.to_string()))?;
        }

        let buffer_end = self.buffer_start_sample + self.buffer.len() as u64;
        self.drop_before(buffer_end);
        Ok(out)
    }
}

pub(crate) fn duration_from_samples(samples: u64) -> std::time::Duration {
    let nanos = samples as u128 * 1_000_000_000u128 / PCM_SAMPLE_RATE_HZ as u128;
    std::time::Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}
