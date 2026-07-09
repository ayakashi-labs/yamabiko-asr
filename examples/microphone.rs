mod common;

use common::audio::{ASR_CHUNK_SAMPLES, MicResampler, TARGET_SAMPLE_RATE, downmix_to_mono, rms};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::time::Instant;
use yamabiko_asr::{PcmChunk, Transcriber};

const USAGE: &str = "usage: microphone [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]";

#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;
    let config = common::transcriber_config(&args)?;
    eprintln!("[asr] model dir: {}", config.model_dir.display());
    eprintln!("[asr] device: {}", config.device);
    eprintln!("[asr] language: {:?}", config.language);
    eprintln!("[asr] vad: {:?}", config.vad);

    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("no input device")?;
    let supported = device.default_input_config()?;
    eprintln!("[asr] input config: {supported:?}");
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "this example expects f32 input, got {:?}; convert to f32 mono 16 kHz in production",
            supported.sample_format()
        )
        .into());
    }

    let input_channels = supported.channels() as usize;
    let input_sample_rate = supported.sample_rate();
    if input_channels > 1 {
        eprintln!("[asr] downmixing {input_channels} channels -> mono");
    }
    if input_sample_rate != TARGET_SAMPLE_RATE {
        eprintln!("[asr] resampling {input_sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz");
    }

    let stream_config: cpal::StreamConfig = supported.into();
    let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    eprintln!("[asr] opening input stream...");
    let stream = device.build_input_stream(
        stream_config,
        move |data: &[f32], _| {
            let samples = downmix_to_mono(data, input_channels);
            let _ = pcm_tx.send(samples);
        },
        move |err| eprintln!("input stream error: {err}"),
        None,
    )?;

    eprintln!("[asr] loading model...");
    let started = Instant::now();
    let transcriber = Transcriber::new(config)?;
    eprintln!("[asr] model loaded in {:.2?}", started.elapsed());
    let (input, mut events) = transcriber.start().into_channels();
    std::thread::spawn(move || {
        let mut resampler = match MicResampler::new(input_sample_rate) {
            Ok(resampler) => resampler,
            Err(err) => {
                eprintln!("[asr] failed to create resampler: {err}");
                return;
            }
        };
        let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);
        let mut chunk_count = 0usize;
        let mut input_sample_count = 0usize;
        let mut output_sample_count = 0usize;

        while let Ok(samples) = pcm_rx.recv() {
            chunk_count += 1;
            input_sample_count += samples.len();
            if chunk_count == 1 || chunk_count.is_multiple_of(50) {
                eprintln!(
                    "[asr] received mic chunk #{chunk_count} (input samples: {input_sample_count}, rms: {:.5})",
                    rms(&samples)
                );
            }

            let chunks = match resampler.push(&samples) {
                Ok(chunks) => chunks,
                Err(err) => {
                    eprintln!("[asr] resampling failed: {err}");
                    break;
                }
            };

            for chunk in chunks {
                output_sample_count += chunk.len();
                asr_buffer.extend(chunk);
                while asr_buffer.len() >= ASR_CHUNK_SAMPLES {
                    let remainder = asr_buffer.split_off(ASR_CHUNK_SAMPLES);
                    let chunk = std::mem::replace(&mut asr_buffer, remainder);
                    if input.blocking_send(PcmChunk::new(chunk)).is_err() {
                        return;
                    }
                }
            }

            if chunk_count == 1 || chunk_count.is_multiple_of(50) {
                eprintln!("[asr] forwarded 16 kHz samples: {output_sample_count}");
            }
        }

        if let Ok(chunks) = resampler.finish() {
            for chunk in chunks {
                asr_buffer.extend(chunk);
            }
        }
        if !asr_buffer.is_empty() {
            let _ = input.blocking_send(PcmChunk::new(asr_buffer));
        }
    });

    stream.play()?;
    eprintln!("[asr] listening; speak into the microphone, press Ctrl+C to stop");

    while let Some(event) = events.recv().await {
        if !common::print_segment(event?) {
            eprintln!("[asr] end of stream");
            break;
        }
    }

    Ok(())
}
