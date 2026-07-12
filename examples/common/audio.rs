#![allow(dead_code)]

use super::ExampleResult;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::collections::VecDeque;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const ASR_CHUNK_SAMPLES: usize = 1_600;

pub struct AudioResampler {
    inner: Option<Fft<f32>>,
    pending: VecDeque<f32>,
    input_sample_rate: u32,
    delay_frames_remaining: usize,
    total_input_frames: usize,
    total_output_frames: usize,
}

impl AudioResampler {
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

        let delay_frames_remaining = inner
            .as_ref()
            .map(Resampler::output_delay)
            .unwrap_or_default();

        Ok(Self {
            inner,
            pending: VecDeque::new(),
            input_sample_rate,
            delay_frames_remaining,
            total_input_frames: 0,
            total_output_frames: 0,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> ExampleResult<Vec<Vec<f32>>> {
        if self.inner.is_none() {
            return Ok(vec![samples.to_vec()]);
        }

        self.total_input_frames = self
            .total_input_frames
            .checked_add(samples.len())
            .ok_or("input sample count overflow")?;

        self.pending.extend(samples.iter().copied());
        let mut raw_outputs = Vec::new();
        {
            let resampler = self.inner.as_mut().expect("resampler checked above");
            while self.pending.len() >= resampler.input_frames_next() {
                let input_len = resampler.input_frames_next();
                let input = self.pending.drain(..input_len).collect::<Vec<_>>();
                let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
                let output = resampler.process(&input_adapter, 0, None)?;
                raw_outputs.push(output.take_data());
            }
        }

        let out = raw_outputs
            .into_iter()
            .filter_map(|output| self.prepare_output(output, None))
            .collect();
        Ok(out)
    }

    pub fn finish(&mut self) -> ExampleResult<Vec<Vec<f32>>> {
        if self.inner.is_none() {
            if self.pending.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![self.pending.drain(..).collect()]);
        }

        if self.total_input_frames == 0 {
            return Ok(Vec::new());
        }

        let expected_output_frames = self
            .total_input_frames
            .checked_mul(TARGET_SAMPLE_RATE as usize)
            .and_then(|frames| frames.checked_add(self.input_sample_rate as usize - 1))
            .ok_or("output sample count overflow")?
            / self.input_sample_rate as usize;
        let final_input = self.pending.drain(..).collect::<Vec<_>>();
        let mut first = true;
        let mut out = Vec::new();

        while self.total_output_frames < expected_output_frames {
            let partial_len = if first { final_input.len() } else { 0 };
            let input = if first { final_input.as_slice() } else { &[] };
            let output = {
                let resampler = self.inner.as_mut().expect("resampler checked above");
                process_partial(resampler, input, partial_len)?
            };
            if output.is_empty() {
                return Err("resampler did not produce the expected final samples".into());
            }
            if let Some(output) = self.prepare_output(output, Some(expected_output_frames)) {
                out.push(output);
            }
            first = false;
        }

        Ok(out)
    }

    fn prepare_output(
        &mut self,
        mut output: Vec<f32>,
        expected_output_frames: Option<usize>,
    ) -> Option<Vec<f32>> {
        let trim = self.delay_frames_remaining.min(output.len());
        if trim > 0 {
            output.drain(..trim);
            self.delay_frames_remaining -= trim;
        }

        if let Some(expected) = expected_output_frames {
            output.truncate(expected.saturating_sub(self.total_output_frames));
        }
        self.total_output_frames += output.len();

        (!output.is_empty()).then_some(output)
    }
}

fn process_partial(
    resampler: &mut Fft<f32>,
    input: &[f32],
    partial_len: usize,
) -> ExampleResult<Vec<f32>> {
    let dummy = [0.0];
    let adapter_input = if input.is_empty() { &dummy[..] } else { input };
    let input_adapter = InterleavedSlice::new(adapter_input, 1, adapter_input.len())?;
    let mut output = vec![0.0; resampler.output_frames_max().max(1)];
    let out_capacity = output.len();
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, out_capacity)?;
    let indexing = rubato::Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: Some(partial_len),
    };
    let (_, frames_written) =
        resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))?;
    output.truncate(frames_written);
    Ok(output)
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

#[cfg(target_os = "windows")]
pub fn wasapi_capture_time(session_started: Instant, info: &cpal::InputCallbackInfo) -> Duration {
    let timestamp = info.timestamp();
    let capture_delay = timestamp.callback.duration_since(timestamp.capture);
    session_started.elapsed().saturating_sub(capture_delay)
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}
