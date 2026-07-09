# asr-crate

Parakeet-family on-device transcription crate for desktop apps. The current
implementation targets f32 mono 16 kHz PCM, runs Silero VAD before ASR, and
uses a local ONNX runner for Parakeet TDT models.

## Current Scope

- Tokio-based streaming input/output API.
- Input timestamps are preserved even when VAD removes silent audio before ASR.
- Output events currently contain VAD-final utterance segments.
- CPU and DirectML execution can be selected explicitly.
- Current model path: `nvidia/parakeet-tdt_ctc-0.6b-ja` exported to ONNX.
- Audio capture, resampling, downmixing, and model download are application
  responsibilities.

## Minimal Shape

```rust,no_run
use asr_crate::{PcmChunk, TranscriptEvent, Transcriber, TranscriberConfig};

# async fn run() -> asr_crate::Result<()> {
let config = TranscriberConfig::new("path/to/parakeet-tdt-model");
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

For the Japanese Parakeet TDT model, export ONNX files first:

```powershell
python tools/export_parakeet_tdt_ja.py
cargo run --example microphone -- --vad-min-silence-ms 800 .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

The local ONNX runner is used because `nvidia/parakeet-tdt_ctc-0.6b-ja`
expects 80 mel features.

See `docs/requirements.md` for the current requirements and future scope.
