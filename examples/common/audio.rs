pub use crate::resampler::AudioResampler;
use std::time::{Duration, Instant};

pub const TARGET_SAMPLE_RATE: u32 = yamabiko_asr::PCM_SAMPLE_RATE_HZ;
pub const ASR_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize / 10;

pub fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }

    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub fn wasapi_capture_time(session_started: Instant, info: &cpal::InputCallbackInfo) -> Duration {
    let timestamp = info.timestamp();
    let capture_delay = timestamp.callback.duration_since(timestamp.capture);
    session_started.elapsed().saturating_sub(capture_delay)
}
