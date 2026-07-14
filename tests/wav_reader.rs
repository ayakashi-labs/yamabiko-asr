mod common {
    pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
}

#[path = "../examples/common/wav.rs"]
mod wav;

use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::PathBuf;
use wav::WavPcmReader;

#[test]
fn normalizes_integer_pcm_using_its_declared_bit_depth() {
    for bits_per_sample in [8, 16, 24, 32] {
        let path = temp_wav_path(bits_per_sample);
        write_half_scale_sample(&path, bits_per_sample);

        let mut reader = WavPcmReader::open(&path).unwrap();
        let samples = reader.read_chunk(8).unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(samples.len(), 1);
        assert!(
            (samples[0] - 0.5).abs() < f32::EPSILON,
            "unexpected {bits_per_sample}-bit sample: {}",
            samples[0]
        );
    }
}

fn temp_wav_path(bits_per_sample: u16) -> PathBuf {
    std::env::temp_dir().join(format!(
        "yamabiko-asr-wav-test-{}-{bits_per_sample}.wav",
        std::process::id()
    ))
}

fn write_half_scale_sample(path: &PathBuf, bits_per_sample: u16) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: yamabiko_asr::PCM_SAMPLE_RATE_HZ,
        bits_per_sample,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    match bits_per_sample {
        8 => writer.write_sample(64i8).unwrap(),
        16 => writer.write_sample(16_384i16).unwrap(),
        24 => writer.write_sample(4_194_304i32).unwrap(),
        32 => writer.write_sample(1_073_741_824i32).unwrap(),
        _ => unreachable!(),
    }
    writer.finalize().unwrap();
}
