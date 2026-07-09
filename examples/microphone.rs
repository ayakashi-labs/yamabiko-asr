use asr_crate::{BackendKind, Language, PcmChunk, Transcriber, TranscriberConfig, TranscriptEvent};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::collections::VecDeque;
use std::error::Error;
use std::time::{Duration, Instant};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const ASR_CHUNK_SAMPLES: usize = 1_600;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = parse_args()?;

    let mut config = TranscriberConfig::new(&args.model_dir);
    config.backend = args.backend;
    apply_vad_args(&mut config, &args);
    if let Some(language) = args.language {
        config.language = Language::hint(language)?;
    }
    eprintln!("[asr] model dir: {}", config.model_dir.display());
    eprintln!("[asr] backend: {}", config.backend);
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
        match event? {
            TranscriptEvent::Segment(segment) => {
                let state = if segment.is_final { "final" } else { "partial" };
                eprintln!(
                    "[asr] backend emitted {state} text ({:.2?}..{:.2?})",
                    segment.start, segment.end
                );
                println!("[{}] {}", state, segment.text);
            }
            TranscriptEvent::EndOfStream => {
                eprintln!("[asr] end of stream");
                break;
            }
        }
    }

    Ok(())
}

struct ExampleArgs {
    model_dir: String,
    backend: BackendKind,
    language: Option<String>,
    vad_threshold: Option<f32>,
    vad_min_speech_ms: Option<u64>,
    vad_min_silence_ms: Option<u64>,
    vad_speech_pad_ms: Option<u64>,
}

fn parse_args() -> Result<ExampleArgs, Box<dyn Error + Send + Sync>> {
    let mut backend = BackendKind::default();
    let mut vad_threshold = None;
    let mut vad_min_speech_ms = None;
    let mut vad_min_silence_ms = None;
    let mut vad_speech_pad_ms = None;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--backend" {
            let value = args.next().ok_or("missing value for --backend")?;
            backend = value.parse()?;
        } else if let Some(value) = arg.strip_prefix("--backend=") {
            backend = value.parse()?;
        } else if arg == "--vad-threshold" {
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

    if positional.is_empty() || positional.len() > 2 {
        return Err(
            "usage: microphone [--backend nemotron|parakeet-tdt] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> [language]"
                .into(),
        );
    }

    Ok(ExampleArgs {
        model_dir: positional.remove(0),
        backend,
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

struct MicResampler {
    inner: Option<Fft<f32>>,
    pending: VecDeque<f32>,
}

impl MicResampler {
    fn new(input_sample_rate: u32) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let inner = if input_sample_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(Fft::<f32>::new(
                input_sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
                (input_sample_rate as usize / 100).max(1),
                2,
                1,
                FixedSync::Input,
            )?)
        };

        Ok(Self {
            inner,
            pending: VecDeque::new(),
        })
    }

    fn push(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let Some(resampler) = self.inner.as_mut() else {
            return Ok(vec![samples.to_vec()]);
        };

        self.pending.extend(samples.iter().copied());
        let mut out = Vec::new();
        while self.pending.len() >= resampler.input_frames_next() {
            let input_len = resampler.input_frames_next();
            let input = self.pending.drain(..input_len).collect::<Vec<_>>();
            let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
            let output = resampler.process(&input_adapter, 0, None)?;
            out.push(output.take_data());
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let Some(resampler) = self.inner.as_mut() else {
            if self.pending.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![self.pending.drain(..).collect()]);
        };

        let input_len = self.pending.len();
        if input_len == 0 {
            return Ok(Vec::new());
        }

        let input = self.pending.drain(..).collect::<Vec<_>>();
        let input_adapter = InterleavedSlice::new(&input, 1, input_len)?;
        let mut output = vec![0.0; resampler.output_frames_next()];
        let out_capacity = output.len();
        let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, out_capacity)?;
        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: None,
            partial_len: Some(input_len),
        };
        let (_, frames_written) =
            resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))?;
        output.truncate(frames_written);
        Ok(vec![output])
    }
}

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }

    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}
