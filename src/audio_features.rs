use ndarray::Array2;
use realfft::{RealToComplex, num_complex::Complex32};

const MEL_LINEAR_SCALE: f64 = 200.0 / 3.0;
const MEL_LOG_FREQUENCY: f64 = 1_000.0;
const MEL_LOG_VALUE: f64 = MEL_LOG_FREQUENCY / MEL_LINEAR_SCALE;
const MEL_LOG_STEP: f64 = 0.06875177742094912;

pub(crate) struct StftWorkspace {
    input: Vec<f32>,
    output: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl StftWorkspace {
    pub(crate) fn new(plan: &dyn RealToComplex<f32>) -> Self {
        Self {
            input: plan.make_input_vec(),
            output: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
        }
    }

    pub(crate) fn execute_centered(
        &mut self,
        plan: &dyn RealToComplex<f32>,
        window: &[f32],
        center: i128,
        mut sample_at: impl FnMut(i128) -> f32,
    ) -> std::result::Result<&[Complex32], String> {
        if window.len() > self.input.len() {
            return Err(format!(
                "STFT window length {} exceeds FFT length {}",
                window.len(),
                self.input.len()
            ));
        }

        self.input.fill(0.0);
        let window_offset = (self.input.len() - window.len()) / 2;
        let frame_start = center - (window.len() / 2) as i128;
        for (window_index, weight) in window.iter().copied().enumerate() {
            self.input[window_offset + window_index] =
                sample_at(frame_start + window_index as i128) * weight;
        }
        plan.process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch)
            .map_err(|err| err.to_string())?;
        Ok(&self.output)
    }
}

pub(crate) fn hann_window(window_length: usize) -> Vec<f32> {
    match window_length {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..window_length)
            .map(|index| {
                0.5 - 0.5
                    * ((2.0 * std::f32::consts::PI * index as f32) / (window_length as f32 - 1.0))
                        .cos()
            })
            .collect(),
    }
}

pub(crate) fn slaney_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
    minimum_frequency: f64,
    maximum_frequency: f64,
) -> Array2<f32> {
    let frequency_bins = n_fft / 2 + 1;
    let mel_minimum = hz_to_mel_slaney(minimum_frequency);
    let mel_maximum = hz_to_mel_slaney(maximum_frequency);
    let mel_points = (0..=n_mels + 1)
        .map(|index| {
            mel_to_hz_slaney(
                mel_minimum + (mel_maximum - mel_minimum) * index as f64 / (n_mels + 1) as f64,
            )
        })
        .collect::<Vec<_>>();
    let fft_frequencies = (0..frequency_bins)
        .map(|index| index as f64 * sample_rate as f64 / n_fft as f64)
        .collect::<Vec<_>>();
    let differences = mel_points
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();
    let mut basis = Array2::<f32>::zeros((n_mels, frequency_bins));

    for mel in 0..n_mels {
        for (frequency, value) in fft_frequencies.iter().copied().enumerate() {
            let lower = (value - mel_points[mel]) / differences[mel];
            let upper = (mel_points[mel + 2] - value) / differences[mel + 1];
            basis[[mel, frequency]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }
    for mel in 0..n_mels {
        let normalization = (2.0 / (mel_points[mel + 2] - mel_points[mel])) as f32;
        for frequency in 0..frequency_bins {
            basis[[mel, frequency]] *= normalization;
        }
    }
    basis
}

fn hz_to_mel_slaney(frequency: f64) -> f64 {
    if frequency < MEL_LOG_FREQUENCY {
        frequency / MEL_LINEAR_SCALE
    } else {
        MEL_LOG_VALUE + (frequency / MEL_LOG_FREQUENCY).ln() / MEL_LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MEL_LOG_VALUE {
        MEL_LINEAR_SCALE * mel
    } else {
        MEL_LOG_FREQUENCY * (MEL_LOG_STEP * (mel - MEL_LOG_VALUE)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_handles_degenerate_lengths() {
        assert!(hann_window(0).is_empty());
        assert_eq!(hann_window(1), [1.0]);
    }

    #[test]
    fn centered_stft_zero_pads_outside_the_audio_timeline() {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(8);
        let mut workspace = StftWorkspace::new(plan.as_ref());
        let audio = [1.0, 0.0, 0.0, 0.0];
        let output = workspace
            .execute_centered(plan.as_ref(), &[1.0, 2.0, 3.0, 4.0], 0, |sample| {
                usize::try_from(sample)
                    .ok()
                    .and_then(|index| audio.get(index))
                    .copied()
                    .unwrap_or(0.0)
            })
            .unwrap();

        for value in output {
            assert!((value.norm_sqr() - 9.0).abs() < 1e-5);
        }
    }

    #[test]
    fn slaney_filterbank_has_expected_shape_and_finite_values() {
        let filterbank = slaney_mel_filterbank(512, 80, 16_000, 0.0, 8_000.0);
        assert_eq!(filterbank.shape(), [80, 257]);
        assert!(filterbank.iter().all(|value| value.is_finite()));
    }
}
