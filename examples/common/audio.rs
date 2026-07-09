#![allow(dead_code)]

use super::ExampleResult;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::collections::VecDeque;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const ASR_CHUNK_SAMPLES: usize = 1_600;

pub struct MicResampler {
    inner: Option<Fft<f32>>,
    pending: VecDeque<f32>,
}

impl MicResampler {
    pub fn new(input_sample_rate: u32) -> ExampleResult<Self> {
        let inner = if input_sample_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(Fft::<f32>::new(
                input_sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
                (input_sample_rate as usize / 100).max(1),
                2,
                1,
                FixedSync::Input,
            )?)
        };

        Ok(Self {
            inner,
            pending: VecDeque::new(),
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> ExampleResult<Vec<Vec<f32>>> {
        let Some(resampler) = self.inner.as_mut() else {
            return Ok(vec![samples.to_vec()]);
        };

        self.pending.extend(samples.iter().copied());
        let mut out = Vec::new();
        while self.pending.len() >= resampler.input_frames_next() {
            let input_len = resampler.input_frames_next();
            let input = self.pending.drain(..input_len).collect::<Vec<_>>();
            let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
            let output = resampler.process(&input_adapter, 0, None)?;
            out.push(output.take_data());
        }
        Ok(out)
    }

    pub fn finish(&mut self) -> ExampleResult<Vec<Vec<f32>>> {
        let Some(resampler) = self.inner.as_mut() else {
            if self.pending.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![self.pending.drain(..).collect()]);
        };

        let input_len = self.pending.len();
        if input_len == 0 {
            return Ok(Vec::new());
        }

        let input = self.pending.drain(..).collect::<Vec<_>>();
        let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
        let mut output = vec![0.0; resampler.output_frames_next()];
        let out_capacity = output.len();
        let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, out_capacity)?;
        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: None,
            partial_len: Some(input_len),
        };
        let (_, frames_written) =
            resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))?;
        output.truncate(frames_written);
        Ok(vec![output])
    }
}

pub fn read_wav_mono_16k(path: &str) -> ExampleResult<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != TARGET_SAMPLE_RATE {
        return Err("expected mono 16 kHz WAV; resample/downmix before using this crate".into());
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
    };

    Ok(samples)
}

pub fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }

    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}
