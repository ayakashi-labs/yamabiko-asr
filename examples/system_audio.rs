mod common;

#[cfg(target_os = "windows")]
use common::audio::{
    ASR_CHUNK_SAMPLES, AudioResampler, TARGET_SAMPLE_RATE, downmix_to_mono, wasapi_capture_time,
};
#[cfg(target_os = "windows")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};
#[cfg(target_os = "windows")]
use yamabiko_asr::{Language, PcmChunk, Transcriber};

const USAGE: &str = "usage: system_audio [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]";

#[cfg(target_os = "windows")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;
    let config = common::transcriber_config(&args)?;

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let output_device_name = device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "Unknown output device".to_string());
    let supported = device.default_output_config()?;
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "this example expects f32 loopback audio, got {:?}; convert the callback samples to f32 in production",
            supported.sample_format()
        )
        .into());
    }

    let output_channels = supported.channels() as usize;
    let output_sample_rate = supported.sample_rate();
    let output_sample_format = supported.sample_format();
    let execution = config.device;
    let language = match &config.language {
        Language::Auto => "auto".to_string(),
        Language::Hint(hint) => hint.clone(),
        _ => "unknown".to_string(),
    };

    let started = Instant::now();
    let transcriber = Transcriber::new(config)?;
    println!(
        "[{}] Model loaded in {:.2}s",
        common::local_time(),
        started.elapsed().as_secs_f64()
    );

    let session_started = Instant::now();
    let (input, mut events, worker) = transcriber.start().into_parts();
    let stream_config: cpal::StreamConfig = supported.into();
    let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<(Duration, Vec<f32>)>();
    let stream = device.build_input_stream(
        stream_config,
        move |data: &[f32], info| {
            let captured_at = wasapi_capture_time(session_started, info);
            let samples = downmix_to_mono(data, output_channels);
            let _ = pcm_tx.send((captured_at, samples));
        },
        move |err| eprintln!("loopback stream error: {err}"),
        None,
    )?;

    println!("  Output device {output_device_name}");
    if output_channels == 1 && output_sample_rate == TARGET_SAMPLE_RATE {
        println!("  Loopback 1 ch / {TARGET_SAMPLE_RATE} Hz / {output_sample_format:?}");
    } else {
        println!(
            "  Loopback {output_channels} ch / {output_sample_rate} Hz / {output_sample_format:?} -> ASR mono / {TARGET_SAMPLE_RATE} Hz / F32"
        );
    }
    println!("  Execution {execution} / Language {language}");

    let audio_forwarder = std::thread::spawn(move || {
        let mut resampler = match AudioResampler::new(output_sample_rate) {
            Ok(resampler) => resampler,
            Err(err) => {
                eprintln!("failed to create resampler: {err}");
                return;
            }
        };
        let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);
        let mut source_started_at = None;
        let mut timeline_anchored = false;

        while let Ok((captured_at, samples)) = pcm_rx.recv() {
            if !timeline_anchored {
                source_started_at.get_or_insert(captured_at);
            }
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
                    let result = if !timeline_anchored {
                        let timestamp = source_started_at.take().unwrap_or(Duration::ZERO);
                        let result = input.blocking_send_at(timestamp, PcmChunk::new(chunk));
                        if result.is_ok() {
                            timeline_anchored = true;
                        }
                        result
                    } else {
                        input.blocking_send(PcmChunk::new(chunk))
                    };
                    if result.is_err() {
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
            if !timeline_anchored {
                let timestamp = source_started_at.take().unwrap_or(Duration::ZERO);
                let _ = input.blocking_send_at(timestamp, PcmChunk::new(asr_buffer));
            } else {
                let _ = input.blocking_send(PcmChunk::new(asr_buffer));
            }
        }
        let _ = input.blocking_close();
    });

    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })?;

    stream.play()?;
    println!("[{}] Capturing system audio...", common::local_time());
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

#[cfg(not(target_os = "windows"))]
fn main() -> common::ExampleResult<()> {
    Err("system_audio uses WASAPI loopback and currently supports only Windows".into())
}
