use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::collections::VecDeque;
use std::error::Error;

type AudioResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const TARGET_SAMPLE_RATE: u32 = yamabiko_asr::PCM_SAMPLE_RATE_HZ;

pub struct AudioResampler {
    inner: Option<Fft<f32>>,
    pending: VecDeque<f32>,
    input_sample_rate: u32,
    delay_frames_remaining: usize,
    total_input_frames: usize,
    total_output_frames: usize,
}

impl AudioResampler {
    pub fn new(input_sample_rate: u32) -> AudioResult<Self> {
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

    pub fn push(&mut self, samples: &[f32], output: &mut Vec<f32>) -> AudioResult<()> {
        if self.inner.is_none() {
            output.extend_from_slice(samples);
            return Ok(());
        }

        self.total_input_frames = self
            .total_input_frames
            .checked_add(samples.len())
            .ok_or("input sample count overflow")?;

        self.pending.extend(samples.iter().copied());
        loop {
            let raw_output = {
                let resampler = self.inner.as_mut().expect("resampler checked above");
                if self.pending.len() < resampler.input_frames_next() {
                    break;
                }
                let input_len = resampler.input_frames_next();
                let input = self.pending.drain(..input_len).collect::<Vec<_>>();
                let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
                resampler.process(&input_adapter, 0, None)?.take_data()
            };
            self.append_output(raw_output, None, output);
        }

        Ok(())
    }

    pub fn finish(&mut self, output: &mut Vec<f32>) -> AudioResult<()> {
        if self.inner.is_none() {
            return Ok(());
        }

        if self.total_input_frames == 0 {
            return Ok(());
        }

        let expected_output_frames = self
            .total_input_frames
            .checked_mul(TARGET_SAMPLE_RATE as usize)
            .and_then(|frames| frames.checked_add(self.input_sample_rate as usize - 1))
            .ok_or("output sample count overflow")?
            / self.input_sample_rate as usize;
        let final_input = self.pending.drain(..).collect::<Vec<_>>();
        let mut first = true;

        while self.total_output_frames < expected_output_frames {
            let partial_len = if first { final_input.len() } else { 0 };
            let input = if first { final_input.as_slice() } else { &[] };
            let resampled = {
                let resampler = self.inner.as_mut().expect("resampler checked above");
                process_partial(resampler, input, partial_len)?
            };
            if resampled.is_empty() {
                return Err("resampler did not produce the expected final samples".into());
            }
            self.append_output(resampled, Some(expected_output_frames), output);
            first = false;
        }

        Ok(())
    }

    fn append_output(
        &mut self,
        mut resampled: Vec<f32>,
        expected_output_frames: Option<usize>,
        output: &mut Vec<f32>,
    ) {
        let trim = self.delay_frames_remaining.min(resampled.len());
        if trim > 0 {
            resampled.drain(..trim);
            self.delay_frames_remaining -= trim;
        }

        if let Some(expected) = expected_output_frames {
            resampled.truncate(expected.saturating_sub(self.total_output_frames));
        }
        self.total_output_frames += resampled.len();
        output.extend(resampled);
    }
}

fn process_partial(
    resampler: &mut Fft<f32>,
    input: &[f32],
    partial_len: usize,
) -> AudioResult<Vec<f32>> {
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
