#![allow(dead_code)]

use asr_crate::{Device, Language, TranscriberConfig, TranscriptEvent};
use std::error::Error;
use std::time::Duration;

pub mod audio;

pub type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

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
    let mut config = TranscriberConfig::new(&args.model_dir);
    if let Some(device) = args.device {
        config.device = device;
    }
    if let Some(language) = &args.language {
        config.language = Language::hint(language)?;
    }
    if let Some(threshold) = args.vad_threshold {
        config.vad.threshold = threshold;
    }
    if let Some(ms) = args.vad_min_speech_ms {
        config.vad.min_speech = Duration::from_millis(ms);
    }
    if let Some(ms) = args.vad_min_silence_ms {
        config.vad.min_silence = Duration::from_millis(ms);
    }
    if let Some(ms) = args.vad_speech_pad_ms {
        config.vad.speech_pad = Duration::from_millis(ms);
    }
    Ok(config)
}

pub fn print_segment(event: TranscriptEvent) -> bool {
    match event {
        TranscriptEvent::Segment(segment) => {
            let state = if segment.is_final { "final" } else { "partial" };
            eprintln!(
                "[asr] backend emitted {state} text ({:.2?}..{:.2?})",
                segment.start, segment.end
            );
            println!("[{}] {}", state, segment.text);
            true
        }
        TranscriptEvent::EndOfStream => false,
    }
}

pub fn print_timed_segment(event: TranscriptEvent) -> bool {
    match event {
        TranscriptEvent::Segment(segment) => {
            let state = if segment.is_final { "final" } else { "partial" };
            println!(
                "[{} {:.2?}..{:.2?}] {}",
                state, segment.start, segment.end, segment.text
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
