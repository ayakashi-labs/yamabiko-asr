# asr-crate

Multilingual streaming transcription crate for desktop apps. The current
implementation targets f32 mono 16 kHz PCM, runs Silero VAD before ASR, and
uses `parakeet-rs` Nemotron streaming models for on-device transcription.
It can also run Parakeet TDT models as a final-only utterance backend.

## Current Scope

- Tokio-based streaming API.
- Input timestamps are preserved even when VAD removes silent audio before ASR.
- Output events contain partial/final transcript segments.
- CPU and DirectML execution can be selected explicitly.
- Backends: `nemotron` for true streaming, `parakeet-tdt` for VAD-final
  Japanese utterance recognition with 80 mel TDT models.
- Audio capture, resampling, downmixing, and model download are application
  responsibilities.

## Minimal Shape

```rust,no_run
use asr_crate::{PcmChunk, TranscriptEvent, Transcriber, TranscriberConfig};

# async fn run() -> asr_crate::Result<()> {
let config = TranscriberConfig::new("path/to/nemotron-model");
let transcriber = Transcriber::new(config)?;
let (input, mut events) = transcriber.start().into_channels();

input.send(PcmChunk::new(vec![0.0; 1600])).await.unwrap();
drop(input);

while let Some(event) = events.recv().await {
    if let TranscriptEvent::Segment(segment) = event? {
        println!("{}", segment.text);
    }
}
# Ok(())
# }
```

For the Japanese Parakeet TDT model, export ONNX files first and select the
backend in examples:

```powershell
python tools/export_parakeet_tdt_ja.py
cargo run --example microphone -- --backend parakeet-tdt --vad-min-silence-ms 800 .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

The `parakeet-tdt` backend uses a small local ONNX runner because the current
`parakeet-rs::ParakeetTDT` helper assumes 128 mel features, while
`nvidia/parakeet-tdt_ctc-0.6b-ja` expects 80.

See `docs/requirements.md` for the current requirements and future scope.
