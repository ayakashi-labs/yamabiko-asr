mod common;

use asr_crate::{PcmChunk, Transcriber};

const USAGE: &str = "usage: wav_stream [--device auto|cpu|directml|cuda|tensorrt|openvino|rocm|coreml|xnnpack|onednn] [--vad-threshold VALUE] [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-speech-pad-ms MS] <model-dir> <16k-mono-wav> [language]";

#[tokio::main(flavor = "current_thread")]
async fn main() -> common::ExampleResult<()> {
    let args = common::parse_args(USAGE, 1)?;
    let config = common::transcriber_config(&args)?;

    let audio = common::audio::read_wav_mono_16k(&args.extra[0])?;
    let transcriber = Transcriber::new(config)?;
    let (input, mut events) = transcriber.start().into_channels();

    let sender = tokio::spawn(async move {
        for chunk in audio.chunks(1_600) {
            input
                .send(PcmChunk::new(chunk.to_vec()))
                .await
                .map_err(|_| "transcription worker closed")?;
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    while let Some(event) = events.recv().await {
        if !common::print_timed_segment(event?) {
            break;
        }
    }

    sender.await??;
    Ok(())
}
