#![cfg(target_os = "windows")]

#[path = "../examples/common/resampler.rs"]
mod resampler;

use resampler::AudioResampler;

fn resample_in_chunks(input_sample_rate: u32, samples: &[f32]) -> Vec<f32> {
    let mut resampler = AudioResampler::new(input_sample_rate).unwrap();
    let mut output = Vec::new();
    for chunk in samples.chunks(137) {
        resampler.push(chunk, &mut output).unwrap();
    }
    resampler.finish(&mut output).unwrap();
    output
}

#[test]
fn preserves_expected_duration_for_partial_chunks() {
    for input_sample_rate in [44_100, 48_000] {
        for input_frames in [1, 137, 439, 440, 441, 479, 480, 481, 4_817] {
            let samples = vec![0.25; input_frames];
            let output = resample_in_chunks(input_sample_rate, &samples);
            let expected_frames = (input_frames * 16_000).div_ceil(input_sample_rate as usize);

            assert_eq!(
                output.len(),
                expected_frames,
                "unexpected duration for {input_frames} frames at {input_sample_rate} Hz"
            );
        }
    }
}

#[test]
fn trims_fft_delay_from_stream_start() {
    for input_sample_rate in [44_100, 48_000] {
        let mut samples = vec![0.0; input_sample_rate as usize / 10];
        samples[0] = 1.0;

        let output = resample_in_chunks(input_sample_rate, &samples);
        let peak = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .unwrap();

        assert_eq!(output.len(), 1_600);
        assert!(
            peak.0 < 5,
            "{input_sample_rate} Hz initial impulse was delayed to frame {}",
            peak.0
        );
    }
}

#[test]
fn flushes_fft_delay_at_stream_end() {
    for input_sample_rate in [44_100, 48_000] {
        let mut samples = vec![0.0; input_sample_rate as usize / 10];
        let final_input_frame = samples.len() - 1;
        samples[final_input_frame] = 1.0;

        let output = resample_in_chunks(input_sample_rate, &samples);
        let peak = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .unwrap();

        assert_eq!(output.len(), 1_600);
        assert!(
            peak.0 >= 1_595,
            "{input_sample_rate} Hz final impulse ended prematurely at frame {}",
            peak.0
        );
    }
}
