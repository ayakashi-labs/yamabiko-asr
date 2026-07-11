#![allow(dead_code)]

use chrono::Local;
use std::error::Error;
use std::time::Duration;
use yamabiko_asr::{Device, Transcriber, TranscriberConfig, TranscriptEvent};

pub mod audio;

pub type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub fn local_time() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

pub struct ExampleArgs {
    pub model_dir: String,
    pub extra: Vec<String>,
    pub language: Option<String>,
    pub device: Option<Device>,
    vad_threshold: Option<f32>,
    vad_min_speech_ms: Option<u64>,
    vad_min_silence_ms: Option<u64>,
    vad_speech_pad_ms: Option<u64>,
}

pub fn parse_args(usage: &str, extra_positionals: usize) -> ExampleResult<ExampleArgs> {
    let mut parsed = ExampleArgs {
        model_dir: String::new(),
        extra: Vec::new(),
        language: None,
        device: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_speech_pad_ms: None,
    };
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => parsed.device = Some(parse_value(&arg, args.next())?),
            "--vad-threshold" => parsed.vad_threshold = Some(parse_value(&arg, args.next())?),
            "--vad-min-speech-ms" => {
                parsed.vad_min_speech_ms = Some(parse_value(&arg, args.next())?)
            }
            "--vad-min-silence-ms" => {
                parsed.vad_min_silence_ms = Some(parse_value(&arg, args.next())?)
            }
            "--vad-speech-pad-ms" => {
                parsed.vad_speech_pad_ms = Some(parse_value(&arg, args.next())?)
            }
            _ if arg.starts_with("--device=") => {
                parsed.device = Some(parse_inline_value(&arg, "--device=")?)
            }
            _ if arg.starts_with("--vad-threshold=") => {
                parsed.vad_threshold = Some(parse_inline_value(&arg, "--vad-threshold=")?)
            }
            _ if arg.starts_with("--vad-min-speech-ms=") => {
                parsed.vad_min_speech_ms = Some(parse_inline_value(&arg, "--vad-min-speech-ms=")?)
            }
            _ if arg.starts_with("--vad-min-silence-ms=") => {
                parsed.vad_min_silence_ms = Some(parse_inline_value(&arg, "--vad-min-silence-ms=")?)
            }
            _ if arg.starts_with("--vad-speech-pad-ms=") => {
                parsed.vad_speech_pad_ms = Some(parse_inline_value(&arg, "--vad-speech-pad-ms=")?)
            }
            _ => positional.push(arg),
        }
    }

    let min_positionals = 1 + extra_positionals;
    let max_positionals = min_positionals + 1;
    if positional.len() < min_positionals || positional.len() > max_positionals {
        return Err(usage.to_string().into());
    }

    parsed.model_dir = positional.remove(0);
    parsed.extra = positional.drain(..extra_positionals).collect();
    parsed.language = positional.pop();
    Ok(parsed)
}

pub fn transcriber_config(args: &ExampleArgs) -> ExampleResult<TranscriberConfig> {
    let mut builder = Transcriber::builder(&args.model_dir);
    if let Some(device) = args.device {
        builder = builder.device(device);
    }
    if let Some(language) = &args.language {
        builder = builder.language_hint(language)?;
    }
    if let Some(threshold) = args.vad_threshold {
        builder = builder.vad_threshold(threshold);
    }
    if let Some(ms) = args.vad_min_speech_ms {
        builder = builder.vad_min_speech(Duration::from_millis(ms));
    }
    if let Some(ms) = args.vad_min_silence_ms {
        builder = builder.vad_min_silence(Duration::from_millis(ms));
    }
    if let Some(ms) = args.vad_speech_pad_ms {
        builder = builder.vad_speech_pad(Duration::from_millis(ms));
    }
    Ok(builder.build_config()?)
}

pub fn print_segment(event: TranscriptEvent) -> bool {
    match event {
        TranscriptEvent::Segment(segment) => {
            let inference_seconds = segment.inference_duration.as_secs_f64();
            let audio_seconds = segment.end.saturating_sub(segment.start).as_secs_f64();
            let rtf = if audio_seconds > 0.0 {
                inference_seconds / audio_seconds
            } else {
                0.0
            };

            println!("[{}] {}", local_time(), segment.text);
            println!(
                "  Inference {inference_seconds:.2}s / Audio {audio_seconds:.2}s / RTF {rtf:.2}"
            );
            true
        }
        TranscriptEvent::EndOfStream => false,
    }
}

fn parse_value<T>(name: &str, value: Option<String>) -> ExampleResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    value
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse()
        .map_err(Into::into)
}

fn parse_inline_value<T>(arg: &str, prefix: &str) -> ExampleResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    arg.strip_prefix(prefix)
        .expect("prefix checked by caller")
        .parse()
        .map_err(Into::into)
}
