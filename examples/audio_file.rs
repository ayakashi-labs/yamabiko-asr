mod common;
#[path = "common/wav.rs"]
mod wav;

use wav::WavPcmReader;
use yamabiko_asr::TranscriptEvent;

const CHUNK_SAMPLES: usize = yamabiko_asr::PCM_SAMPLE_RATE_HZ as usize / 10;

const USAGE: &str = "usage: audio_file [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> <16k-mono-wav>";

#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 1)?;

    let mut audio = WavPcmReader::open(&args.extra[0])?;
    let transcriber = common::load_transcriber(&args)?;
    let (input, mut events, worker) = transcriber.start().into_parts();

    let sender = tokio::spawn(async move {
        while let Some(chunk) = audio.read_chunk(CHUNK_SAMPLES)? {
            input.send(chunk).await?;
        }
        input.close().await?;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    while let Some(event) = events.recv().await {
        match event? {
            TranscriptEvent::Segment(segment) => common::print_transcript(&segment, None),
            TranscriptEvent::EndOfStream => break,
            _ => {}
        }
    }

    sender.await??;
    worker.await?;
    Ok(())
}
