mod common;

use common::audio::{ASR_CHUNK_SAMPLES, AudioResampler, TARGET_SAMPLE_RATE, downmix_to_mono};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::time::Instant;
use yamabiko_asr::{Language, PcmChunk, Transcriber};

const USAGE: &str = "usage: microphone [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]";

#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;
    let config = common::transcriber_config(&args)?;

    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("no input device")?;
    let input_device_name = device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "Unknown input device".to_string());
    let supported = device.default_input_config()?;
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "this example expects f32 input, got {:?}; convert to f32 mono 16 kHz in production",
            supported.sample_format()
        )
        .into());
    }

    let input_channels = supported.channels() as usize;
    let input_sample_rate = supported.sample_rate();
    let input_sample_format = supported.sample_format();
    let execution = config.device;
    let language = match &config.language {
        Language::Auto => "auto".to_string(),
        Language::Hint(hint) => hint.clone(),
        _ => "unknown".to_string(),
    };

    let stream_config: cpal::StreamConfig = supported.into();
    let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let stream = device.build_input_stream(
        stream_config,
        move |data: &[f32], _| {
            let samples = downmix_to_mono(data, input_channels);
            let _ = pcm_tx.send(samples);
        },
        move |err| eprintln!("input stream error: {err}"),
        None,
    )?;

    let started = Instant::now();
    let transcriber = Transcriber::new(config)?;
    println!(
        "[{}] Model loaded in {:.2}s",
        common::local_time(),
        started.elapsed().as_secs_f64()
    );
    println!("  Input device {input_device_name}");
    if input_channels == 1 && input_sample_rate == TARGET_SAMPLE_RATE {
        println!("  Input 1 ch / {TARGET_SAMPLE_RATE} Hz / {input_sample_format:?}");
    } else {
        println!(
            "  Input {input_channels} ch / {input_sample_rate} Hz / {input_sample_format:?} -> ASR mono / {TARGET_SAMPLE_RATE} Hz / F32"
        );
    }
    println!("  Execution {execution} / Language {language}");
    let (input, mut events, worker) = transcriber.start().into_parts();
    let audio_forwarder = std::thread::spawn(move || {
        let mut resampler = match AudioResampler::new(input_sample_rate) {
            Ok(resampler) => resampler,
            Err(err) => {
                eprintln!("failed to create resampler: {err}");
                return;
            }
        };
        let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);

        while let Ok(samples) = pcm_rx.recv() {
            let chunks = match resampler.push(&samples) {
                Ok(chunks) => chunks,
                Err(err) => {
                    eprintln!("resampling failed: {err}");
                    break;
                }
            };

            for chunk in chunks {
                asr_buffer.extend(chunk);
                while asr_buffer.len() >= ASR_CHUNK_SAMPLES {
                    let remainder = asr_buffer.split_off(ASR_CHUNK_SAMPLES);
                    let chunk = std::mem::replace(&mut asr_buffer, remainder);
                    if input.blocking_send(PcmChunk::new(chunk)).is_err() {
                        return;
                    }
                }
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
        let _ = input.blocking_close();
    });

    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })?;

    stream.play()?;
    println!("[{}] Listening...", common::local_time());
    let mut stream = Some(stream);
    let mut stopping = false;

    loop {
        tokio::select! {
            _ = stop_rx.recv(), if !stopping => {
                stopping = true;
                stream.take();
                println!("[{}] Stopping...", common::local_time());
            }
            event = events.recv() => {
                let Some(event) = event else {
                    break;
                };
                if !common::print_segment(event?) {
                    break;
                }
            }
        }
    }

    drop(stream);
    audio_forwarder
        .join()
        .map_err(|_| "audio forwarding thread panicked")?;
    worker.await?;
    println!("[{}] Stopped.", common::local_time());

    Ok(())
}
