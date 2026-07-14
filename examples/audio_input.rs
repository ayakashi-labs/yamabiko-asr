mod common;

#[cfg(target_os = "windows")]
use common::capture::{CaptureDevice, print_event};
#[cfg(target_os = "windows")]
use std::time::Instant;
#[cfg(target_os = "windows")]
use yamabiko_asr::{AudioSourceConfig, Language, Transcriber};

const USAGE: &str = "usage: audio_input [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]";

// Choose one capture mode by commenting out one line and uncommenting the other.
const CAPTURE_SYSTEM_AUDIO: bool = true; // microphone + system audio
// const CAPTURE_SYSTEM_AUDIO: bool = false; // microphone only

#[cfg(target_os = "windows")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 0)?;
    let config = common::transcriber_config(&args)?;

    let host = cpal::default_host();
    let microphone_device = CaptureDevice::microphone(&host)?;
    let system_device = if CAPTURE_SYSTEM_AUDIO {
        Some(CaptureDevice::system_audio(&host)?)
    } else {
        None
    };
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
    let system_input = if CAPTURE_SYSTEM_AUDIO {
        Some(
            session
                .open_source(AudioSourceConfig::system_audio())
                .await?,
        )
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
    println!("  Execution {execution} / Language {language}");

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
    println!("[{}] Transcribing {mode}...", common::local_time());
    let mut stopping = false;

    loop {
        tokio::select! {
            _ = stop_rx.recv(), if !stopping => {
                stopping = true;
                for capture in &mut captures {
                    capture.stop();
                }
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

    for capture in &mut captures {
        capture.stop();
    }
    for capture in captures {
        capture.join()?;
    }
    worker.await?;
    println!("[{}] Stopped.", common::local_time());

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn main() -> common::ExampleResult<()> {
    Err("audio_input currently supports only Windows".into())
}
