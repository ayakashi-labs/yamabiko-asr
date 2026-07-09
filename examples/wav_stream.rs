use asr_crate::{Language, PcmChunk, Transcriber, TranscriberConfig, TranscriptEvent};
use std::error::Error;
use std::time::Duration;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = parse_args()?;

    let mut config = TranscriberConfig::new(&args.model_dir);
    apply_vad_args(&mut config, &args);
    if let Some(language) = args.language {
        config.language = Language::hint(language)?;
    }

    let audio = read_wav_mono_16k(&args.wav_path)?;
    let transcriber = Transcriber::new(config)?;
    let (input, mut events) = transcriber.start().into_channels();

    let sender = tokio::spawn(async move {
        for chunk in audio.chunks(1_600) {
            input
                .send(PcmChunk::new(chunk.to_vec()))
                .await
                .map_err(|_| "transcription worker closed")?;
        }
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    });

    while let Some(event) = events.recv().await {
        match event? {
            TranscriptEvent::Segment(segment) => {
                let state = if segment.is_final { "final" } else { "partial" };
                println!(
                    "[{} {:.2?}..{:.2?}] {}",
                    state, segment.start, segment.end, segment.text
                );
            }
            TranscriptEvent::EndOfStream => break,
        }
    }

    sender.await??;
    Ok(())
}

struct ExampleArgs {
    model_dir: String,
    wav_path: String,
    language: Option<String>,
    vad_threshold: Option<f32>,
    vad_min_speech_ms: Option<u64>,
    vad_min_silence_ms: Option<u64>,
    vad_speech_pad_ms: Option<u64>,
}

fn parse_args() -> Result<ExampleArgs, Box<dyn Error + Send + Sync>> {
    let mut vad_threshold = None;
    let mut vad_min_speech_ms = None;
    let mut vad_min_silence_ms = None;
    let mut vad_speech_pad_ms = None;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--vad-threshold" {
            vad_threshold = Some(parse_f32_arg(&arg, args.next())?);
        } else if let Some(value) = arg.strip_prefix("--vad-threshold=") {
            vad_threshold = Some(value.parse()?);
        } else if arg == "--vad-min-speech-ms" {
            vad_min_speech_ms = Some(parse_u64_arg(&arg, args.next())?);
        } else if let Some(value) = arg.strip_prefix("--vad-min-speech-ms=") {
            vad_min_speech_ms = Some(value.parse()?);
        } else if arg == "--vad-min-silence-ms" {
            vad_min_silence_ms = Some(parse_u64_arg(&arg, args.next())?);
        } else if let Some(value) = arg.strip_prefix("--vad-min-silence-ms=") {
            vad_min_silence_ms = Some(value.parse()?);
        } else if arg == "--vad-speech-pad-ms" {
            vad_speech_pad_ms = Some(parse_u64_arg(&arg, args.next())?);
        } else if let Some(value) = arg.strip_prefix("--vad-speech-pad-ms=") {
            vad_speech_pad_ms = Some(value.parse()?);
        } else {
            positional.push(arg);
        }
    }

    if positional.len() < 2 || positional.len() > 3 {
        return Err(
            "usage: wav_stream [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> <16k-mono-wav> [language]"
                .into(),
        );
    }

    Ok(ExampleArgs {
        model_dir: positional.remove(0),
        wav_path: positional.remove(0),
        language: positional.pop(),
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
    })
}

fn apply_vad_args(config: &mut TranscriberConfig, args: &ExampleArgs) {
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
}

fn parse_f32_arg(name: &str, value: Option<String>) -> Result<f32, Box<dyn Error + Send + Sync>> {
    value
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse()
        .map_err(Into::into)
}

fn parse_u64_arg(name: &str, value: Option<String>) -> Result<u64, Box<dyn Error + Send + Sync>> {
    value
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse()
        .map_err(Into::into)
}

fn read_wav_mono_16k(path: &str) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err("expected mono 16 kHz WAV; resample/downmix before using this crate".into());
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
    };

    Ok(samples)
}
