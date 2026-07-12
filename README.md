# yamabiko-asr

Parakeet-family on-device transcription crate for desktop apps. The current
implementation targets f32 mono 16 kHz PCM, runs Silero VAD before ASR, and
uses a local ONNX runner for Parakeet TDT models.

## Installation

```toml
[dependencies]
yamabiko-asr = "0.1"
```

Enable optional features when needed:

```toml
yamabiko-asr = { version = "0.1", features = ["serde", "directml"] }
```

## Current Scope

- Tokio-based streaming input/output API.
- One loaded ASR model per transcriber. Multiple source streams share that
  model while keeping source audio, VAD state, buffers, and timelines separate
  rather than mixing inputs before transcription.
- Additional sources are registered explicitly and limited by `max_sources`
  (default: 2, including the primary input). Closing one source flushes and
  releases only that source.
- Input timestamps are preserved even when VAD removes silent audio before ASR.
- Output events currently contain VAD-final utterance segments.
- ASR execution device can be selected explicitly: `cpu`, `auto`,
  `directml`, `cuda`, `tensorrt`, `openvino`, `rocm`, `coreml`, `xnnpack`,
  or `onednn`. The default build and default device use CPU. `auto` may try
  enabled accelerators before CPU; explicit accelerator selections require the
  matching Cargo feature and runtime libraries.
- Current verified model path: `nvidia/parakeet-tdt_ctc-0.6b-ja` for Japanese,
  exported to ONNX.
- `nvidia/parakeet-tdt-0.6b-v3` can be exported for experimental multilingual
  testing. In v0.1, run it with automatic language selection; explicit
  non-Japanese language hints are not accepted yet.
- Audio capture, resampling, downmixing, and model download are application
  responsibilities. Examples include small helpers for local testing.

## Minimal Shape

```rust,no_run
use yamabiko_asr::{PcmChunk, TranscriptEvent, Transcriber};

# async fn run() -> yamabiko_asr::Result<()> {
let transcriber = Transcriber::builder("path/to/parakeet-tdt-model").build()?;
let (input, mut events, worker) = transcriber.start().into_parts();

let producer = tokio::spawn(async move {
    input.send(PcmChunk::new(vec![0.0; 1600])).await?;
    input.close().await
});

while let Some(event) = events.recv().await {
    match event? {
        TranscriptEvent::Segment(segment) => {
            println!("{}ms: {}", segment.start_ms(), segment.text);
        }
        TranscriptEvent::EndOfStream => break,
        _ => {}
    }
}

producer
    .await
    .map_err(|_| yamabiko_asr::Error::StreamClosed)??;
worker
    .await
    .map_err(|_| yamabiko_asr::Error::StreamClosed)?;
# Ok(())
# }
```

For Tauri-style UI events, enable the optional `serde` feature and emit the
millisecond payload:

```rust,no_run
# use yamabiko_asr::{TranscriptEvent, TranscriptSegmentPayload};
# fn emit_to_ui(segment: &yamabiko_asr::TranscriptSegment) {
let payload: TranscriptSegmentPayload = segment.to_payload();
// app.emit("transcript-segment", payload)?;
# }
```

Multiple capture streams can share the same loaded ASR model. Register each
additional stream explicitly; closing one input flushes and releases only that
source. Segment timestamps remain source-local and emitted segments carry the
allocated source identifier. Each segment also has a stable `SegmentId`;
consumers should upsert by that ID so later text or speaker revisions can
replace an earlier version:

```rust,no_run
use yamabiko_asr::{AudioSourceConfig, PcmChunk, TranscriptEvent, Transcriber};

# async fn send_audio(
#     transcriber: Transcriber,
# ) -> yamabiko_asr::Result<()> {
let session = transcriber.start();
let system_audio = session
    .open_source(AudioSourceConfig::system_audio())
    .await?;
let (microphone, mut events, worker) = session.into_parts();

let producer = tokio::spawn(async move {
    microphone.send(PcmChunk::new(vec![0.0; 1600])).await?;
    system_audio.send(PcmChunk::new(vec![0.0; 1600])).await?;

    system_audio.close().await?;
    microphone.close().await
});

while let Some(event) = events.recv().await {
    if matches!(event?, TranscriptEvent::EndOfStream) {
        break;
    }
}

producer
    .await
    .map_err(|_| yamabiko_asr::Error::StreamClosed)??;
worker
    .await
    .map_err(|_| yamabiko_asr::Error::StreamClosed)?;
# Ok(())
# }
```

For the Japanese Parakeet TDT model, export ONNX files first:

```powershell
python tools/export_parakeet_tdt_ja.py
cargo run --example microphone -- --vad-min-silence-ms 800 .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

For experimental multilingual Parakeet TDT v3 testing, omit the language
argument and use automatic language selection:

```powershell
python tools/export_parakeet_tdt_multilingual.py
cargo run --example microphone -- --vad-min-silence-ms 800 .\models\parakeet-tdt-0.6b-v3-onnx
```

The local ONNX runner is used because `nvidia/parakeet-tdt_ctc-0.6b-ja`
expects 80 mel features.

The default build uses CPU execution. ONNX Runtime acceleration providers are
opt-in Cargo features, for example:

```powershell
cargo run --features directml --example microphone -- --device directml .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
cargo run --features cuda --example microphone -- --device cuda .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
cargo run --features openvino --example microphone -- --device openvino .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

## License

This crate is licensed under either MIT or Apache-2.0, at your option. Model
files are not distributed by this crate; check each model's license before use.
