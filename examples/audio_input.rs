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

    let mut captures = vec![microphone_device.start(session_started, microphone_input)?];
    if let (Some(device), Some(input)) = (system_device, system_input) {
        captures.push(device.start(session_started, input)?);
    }
    println!("  Execution {execution}");

    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })?;

    for capture in &captures {
        capture.play()?;
    }
    let mode = if CAPTURE_SYSTEM_AUDIO {
        "microphone and system audio"
    } else {
        "microphone"
    };
    println!("Transcribing {mode}...");
    let mut stopping = false;

    loop {
        tokio::select! {
            _ = stop_rx.recv(), if !stopping => {
                stopping = true;
                for capture in &mut captures {
                    capture.stop();
                }
                println!("Stopping...");
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

    for capture in &mut captures {
        capture.stop();
    }
    for capture in captures {
        capture.join()?;
    }
    worker.await?;
    println!("Stopped.");

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn main() -> common::ExampleResult<()> {
    Err("audio_input currently supports only Windows".into())
}
