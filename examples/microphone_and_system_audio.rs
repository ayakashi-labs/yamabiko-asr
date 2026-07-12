mod common;

#[cfg(target_os = "windows")]
use common::audio::{
    ASR_CHUNK_SAMPLES, AudioResampler, TARGET_SAMPLE_RATE, downmix_to_mono, wasapi_capture_time,
};
#[cfg(target_os = "windows")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(target_os = "windows")]
use std::sync::mpsc::Receiver;
#[cfg(target_os = "windows")]
use std::thread::JoinHandle;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};
#[cfg(target_os = "windows")]
use yamabiko_asr::{
    AudioInput, AudioSourceConfig, AudioSourceId, Language, PcmChunk, Transcriber, TranscriptEvent,
};

const USAGE: &str = "usage: microphone_and_system_audio [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]";

#[cfg(target_os = "windows")]
type TimedPcm = (Duration, Vec<f32>);

#[cfg(target_os = "windows")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;
    let config = common::transcriber_config(&args)?;

    let host = cpal::default_host();
    let microphone_device = host.default_input_device().ok_or("no input device")?;
    let system_device = host.default_output_device().ok_or("no output device")?;
    let microphone_name = device_name(&microphone_device);
    let system_name = device_name(&system_device);
    let microphone_supported = microphone_device.default_input_config()?;
    let system_supported = system_device.default_output_config()?;
    require_f32("microphone", microphone_supported.sample_format())?;
    require_f32("system audio", system_supported.sample_format())?;

    let microphone_channels = microphone_supported.channels() as usize;
    let microphone_sample_rate = microphone_supported.sample_rate();
    let microphone_sample_format = microphone_supported.sample_format();
    let system_channels = system_supported.channels() as usize;
    let system_sample_rate = system_supported.sample_rate();
    let system_sample_format = system_supported.sample_format();
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
    let session = transcriber.start();
    let system_input = session
        .open_source(AudioSourceConfig::system_audio())
        .await?;
    let microphone_id = session.input.source_id();
    let system_id = system_input.source_id();
    let (microphone_input, mut events, worker) = session.into_parts();

    let (microphone_tx, microphone_rx) = std::sync::mpsc::channel::<TimedPcm>();
    let microphone_stream = microphone_device.build_input_stream(
        microphone_supported.into(),
        move |data: &[f32], info| {
            let captured_at = wasapi_capture_time(session_started, info);
            let samples = downmix_to_mono(data, microphone_channels);
            let _ = microphone_tx.send((captured_at, samples));
        },
        move |err| eprintln!("microphone stream error: {err}"),
        None,
    )?;

    let (system_tx, system_rx) = std::sync::mpsc::channel::<TimedPcm>();
    let system_stream = system_device.build_input_stream(
        system_supported.into(),
        move |data: &[f32], info| {
            let captured_at = wasapi_capture_time(session_started, info);
            let samples = downmix_to_mono(data, system_channels);
            let _ = system_tx.send((captured_at, samples));
        },
        move |err| eprintln!("system audio loopback error: {err}"),
        None,
    )?;

    print_audio_format(
        "Microphone",
        &microphone_name,
        microphone_channels,
        microphone_sample_rate,
        microphone_sample_format,
    );
    print_audio_format(
        "System audio",
        &system_name,
        system_channels,
        system_sample_rate,
        system_sample_format,
    );
    println!("  Execution {execution} / Language {language}");

    let microphone_forwarder = spawn_forwarder(
        "microphone",
        microphone_input,
        microphone_sample_rate,
        microphone_rx,
    );
    let system_forwarder = spawn_forwarder("system", system_input, system_sample_rate, system_rx);

    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })?;

    microphone_stream.play()?;
    system_stream.play()?;
    println!(
        "[{}] Transcribing microphone and system audio...",
        common::local_time()
    );
    let mut microphone_stream = Some(microphone_stream);
    let mut system_stream = Some(system_stream);
    let mut stopping = false;

    loop {
        tokio::select! {
            _ = stop_rx.recv(), if !stopping => {
                stopping = true;
                microphone_stream.take();
                system_stream.take();
                println!("[{}] Stopping...", common::local_time());
            }
            event = events.recv() => {
                let Some(event) = event else {
                    break;
                };
                if !print_event(event?, microphone_id, system_id) {
                    break;
                }
            }
        }
    }

    drop(microphone_stream);
    drop(system_stream);
    join_forwarder("microphone", microphone_forwarder)?;
    join_forwarder("system audio", system_forwarder)?;
    worker.await?;
    println!("[{}] Stopped.", common::local_time());

    Ok(())
}

#[cfg(target_os = "windows")]
fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "Unknown device".to_string())
}

#[cfg(target_os = "windows")]
fn require_f32(source: &str, format: cpal::SampleFormat) -> common::ExampleResult<()> {
    if format != cpal::SampleFormat::F32 {
        return Err(format!(
            "this example expects f32 {source} audio, got {format:?}; convert the callback samples to f32 in production"
        )
        .into());
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn print_audio_format(
    label: &str,
    device_name: &str,
    channels: usize,
    sample_rate: u32,
    sample_format: cpal::SampleFormat,
) {
    println!("  {label} device {device_name}");
    if channels == 1 && sample_rate == TARGET_SAMPLE_RATE {
        println!("  {label} 1 ch / {TARGET_SAMPLE_RATE} Hz / {sample_format:?}");
    } else {
        println!(
            "  {label} {channels} ch / {sample_rate} Hz / {sample_format:?} -> ASR mono / {TARGET_SAMPLE_RATE} Hz / F32"
        );
    }
}

#[cfg(target_os = "windows")]
fn spawn_forwarder(
    label: &'static str,
    input: AudioInput,
    sample_rate: u32,
    pcm_rx: Receiver<TimedPcm>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(err) = forward_audio(&input, sample_rate, pcm_rx) {
            eprintln!("{label} forwarding failed: {err}");
        }
        if let Err(err) = input.blocking_close() {
            eprintln!("failed to close {label} input: {err}");
        }
    })
}

#[cfg(target_os = "windows")]
fn forward_audio(
    input: &AudioInput,
    sample_rate: u32,
    pcm_rx: Receiver<TimedPcm>,
) -> common::ExampleResult<()> {
    let mut resampler = AudioResampler::new(sample_rate)?;
    let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);
    let mut source_started_at = None;
    let mut timeline_anchored = false;

    while let Ok((captured_at, samples)) = pcm_rx.recv() {
        if !timeline_anchored {
            source_started_at.get_or_insert(captured_at);
        }
        append_resampled(&mut resampler, &mut asr_buffer, &samples)?;
        send_complete_chunks(
            input,
            &mut asr_buffer,
            &mut source_started_at,
            &mut timeline_anchored,
        )?;
    }

    for chunk in resampler.finish()? {
        asr_buffer.extend(chunk);
    }
    if !asr_buffer.is_empty() {
        send_chunk(
            input,
            std::mem::take(&mut asr_buffer),
            &mut source_started_at,
            &mut timeline_anchored,
        )?;
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn append_resampled(
    resampler: &mut AudioResampler,
    asr_buffer: &mut Vec<f32>,
    samples: &[f32],
) -> common::ExampleResult<()> {
    for chunk in resampler.push(samples)? {
        asr_buffer.extend(chunk);
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn send_complete_chunks(
    input: &AudioInput,
    asr_buffer: &mut Vec<f32>,
    source_started_at: &mut Option<Duration>,
    timeline_anchored: &mut bool,
) -> yamabiko_asr::Result<()> {
    while asr_buffer.len() >= ASR_CHUNK_SAMPLES {
        let remainder = asr_buffer.split_off(ASR_CHUNK_SAMPLES);
        let chunk = std::mem::replace(asr_buffer, remainder);
        send_chunk(input, chunk, source_started_at, timeline_anchored)?;
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn send_chunk(
    input: &AudioInput,
    chunk: Vec<f32>,
    source_started_at: &mut Option<Duration>,
    timeline_anchored: &mut bool,
) -> yamabiko_asr::Result<()> {
    if *timeline_anchored {
        input.blocking_send(PcmChunk::new(chunk))
    } else {
        let timestamp = source_started_at.take().unwrap_or(Duration::ZERO);
        input.blocking_send_at(timestamp, PcmChunk::new(chunk))?;
        *timeline_anchored = true;
        Ok(())
    }
}

#[cfg(target_os = "windows")]
fn print_event(
    event: TranscriptEvent,
    microphone_id: AudioSourceId,
    system_id: AudioSourceId,
) -> bool {
    match event {
        TranscriptEvent::Segment(segment) => {
            let source = if segment.source_id == microphone_id {
                "microphone".to_string()
            } else if segment.source_id == system_id {
                "system".to_string()
            } else {
                format!("source:{}", segment.source_id.get())
            };
            let inference_seconds = segment.inference_duration.as_secs_f64();
            let audio_seconds = segment.end.saturating_sub(segment.start).as_secs_f64();
            let rtf = if audio_seconds > 0.0 {
                inference_seconds / audio_seconds
            } else {
                0.0
            };

            println!("[{}] [{source}] {}", common::local_time(), segment.text);
            println!(
                "  Timeline {:.2}-{:.2}s / Inference {inference_seconds:.2}s / Audio {audio_seconds:.2}s / RTF {rtf:.2}",
                segment.start.as_secs_f64(),
                segment.end.as_secs_f64(),
            );
            true
        }
        TranscriptEvent::EndOfStream => false,
        _ => true,
    }
}

#[cfg(target_os = "windows")]
fn join_forwarder(label: &str, forwarder: JoinHandle<()>) -> common::ExampleResult<()> {
    forwarder
        .join()
        .map_err(|_| format!("{label} forwarding thread panicked").into())
}

#[cfg(not(target_os = "windows"))]
fn main() -> common::ExampleResult<()> {
    Err("microphone_and_system_audio uses WASAPI loopback and supports only Windows".into())
}
