#[cfg(target_os = "windows")]
#[path = "common/audio.rs"]
mod audio;
#[cfg(target_os = "windows")]
#[path = "common/capture.rs"]
mod capture;
mod common;
#[cfg(target_os = "windows")]
#[path = "common/resampler.rs"]
mod resampler;

#[cfg(target_os = "windows")]
use capture::{CaptureDevice, print_event};
#[cfg(target_os = "windows")]
use std::time::Instant;
const USAGE: &str = "usage: audio_input [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir>";

// Choose one capture mode by commenting out one line and uncommenting the other.
const CAPTURE_SYSTEM_AUDIO: bool = true; // microphone + system audio
// const CAPTURE_SYSTEM_AUDIO: bool = false; // microphone only

#[cfg(target_os = "windows")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;

    let host = cpal::default_host();
    let microphone_device = CaptureDevice::microphone(&host)?;
    let system_device = if CAPTURE_SYSTEM_AUDIO {
        Some(CaptureDevice::system_audio(&host)?)
    } else {
        None
    };
    let execution = args.device.unwrap_or_default();

    let started = Instant::now();
    let transcriber = common::load_transcriber(&args)?;
    println!("Model loaded in {:.2}s", started.elapsed().as_secs_f64());

    let session_started = Instant::now();
    let session = transcriber.start();
    let system_input = if CAPTURE_SYSTEM_AUDIO {
        Some(session.open_source().await?)
    } else {
        None
    };
    let microphone_id = session.input.source_id();
    let system_id = system_input.as_ref().map(|input| input.source_id());
    let (microphone_input, mut events, worker) = session.into_parts();
    let output_monitor = events.monitor();

    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })?;

    let (capture_failure_tx, mut capture_failure_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut captures = Vec::new();
    let microphone_capture = match microphone_device.start(
        session_started,
        microphone_input,
        capture_failure_tx.clone(),
    ) {
        Ok(capture) => capture,
        Err(error) => {
            events.close();
            worker.await?;
            return Err(error);
        }
    };
    captures.push(microphone_capture);
    if let (Some(device), Some(input)) = (system_device, system_input) {
        match device.start(session_started, input, capture_failure_tx.clone()) {
            Ok(capture) => captures.push(capture),
            Err(error) => {
                events.close();
                let _ = finish_captures(captures);
                worker.await?;
                return Err(error);
            }
        }
    }
    drop(capture_failure_tx);
    println!("  Execution {execution}");

    for capture in &captures {
        if let Err(error) = capture.play() {
            let _ = finish_captures(captures);
            worker.await?;
            return Err(error);
        }
    }
    let mode = if CAPTURE_SYSTEM_AUDIO {
        "microphone and system audio"
    } else {
        "microphone"
    };
    println!("Transcribing {mode}...");
    let mut stopping = false;
    let mut next_backlog_warning = Some(32_usize);
    let mut capture_failure = None;
    let mut capture_failure_channel_open = true;
    let mut transcription_failure = None;

    loop {
        if let Some(threshold) = next_backlog_warning {
            let metrics = output_monitor.metrics();
            if metrics.peak_pending_events >= threshold {
                eprintln!(
                    "Warning: transcript output backlog peaked at {} events (currently {} pending)",
                    metrics.peak_pending_events, metrics.pending_events,
                );

                next_backlog_warning = threshold.checked_mul(2);
                while next_backlog_warning.is_some_and(|next| next <= metrics.peak_pending_events) {
                    next_backlog_warning =
                        next_backlog_warning.and_then(|next| next.checked_mul(2));
                }
            }
        }

        tokio::select! {
            _ = stop_rx.recv(), if !stopping => {
                begin_stopping(&mut captures, &mut stopping);
            }
            failure = capture_failure_rx.recv(), if capture_failure_channel_open && capture_failure.is_none() => {
                match failure {
                    Some(failure) => {
                        eprintln!("{failure}");
                        capture_failure = Some(failure);
                        begin_stopping(&mut captures, &mut stopping);
                    }
                    None => capture_failure_channel_open = false,
                }
            }
            event = events.recv() => {
                match event {
                    Some(Ok(event)) => {
                        if !print_event(event, microphone_id, system_id) {
                            break;
                        }
                    }
                    None => break,
                    Some(Err(error)) => {
                        transcription_failure = Some(error);
                        begin_stopping(&mut captures, &mut stopping);
                        break;
                    }
                }
            }
        }
    }

    let forwarding_failure = finish_captures(captures).err();
    worker.await?;
    let metrics = output_monitor.metrics();
    println!(
        "Output stats: emitted={}, received={}, peak_pending={}, discarded={}, delivery_failures={}",
        metrics.emitted_events,
        metrics.received_events,
        metrics.peak_pending_events,
        metrics.discarded_events,
        metrics.delivery_failures,
    );

    if let Some(error) = transcription_failure {
        return Err(error.into());
    }
    if let Some(error) = forwarding_failure {
        return Err(error);
    }
    if let Some(error) = capture_failure {
        return Err(error.into());
    }
    println!("Stopped.");

    Ok(())
}

#[cfg(target_os = "windows")]
fn begin_stopping(captures: &mut [capture::Capture], stopping: &mut bool) {
    if *stopping {
        return;
    }
    *stopping = true;
    for capture in captures {
        capture.stop();
    }
    println!("Stopping...");
}

#[cfg(target_os = "windows")]
fn finish_captures(mut captures: Vec<capture::Capture>) -> common::ExampleResult<()> {
    for capture in &mut captures {
        capture.stop();
    }
    let mut failures = Vec::new();
    for capture in captures {
        if let Err(error) = capture.join() {
            failures.push(error.to_string());
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

#[cfg(not(target_os = "windows"))]
fn main() -> common::ExampleResult<()> {
    Err("audio_input currently supports only Windows".into())
}
