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
- All sources share one Silero VAD model session while keeping independent VAD
  stream state. Long silence is discarded incrementally, and continuous speech
  is split into utterances of at most 30 seconds by default.
- Sources can be anchored to a shared session timeline with `send_at`; plain
  `send` starts an unanchored source at session time zero and then advances by
  its PCM sample count.
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
  testing. Run it with automatic language selection; explicit
  non-Japanese language hints are not accepted yet.
- Audio capture, resampling, downmixing, and model download are application
  responsibilities. Examples include small helpers for local testing.

## Minimal Shape

```rust,no_run
use yamabiko_asr::{TranscriptEvent, Transcriber};

# async fn run() -> yamabiko_asr::Result<()> {
let transcriber = Transcriber::builder("path/to/parakeet-tdt-model").build()?;
let (input, mut events, worker) = transcriber.start().into_parts();

let producer = tokio::spawn(async move {
    input.send(vec![0.0; 1600]).await?;
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
segment directly. Duration fields serialize as millisecond values named
`start_ms`, `end_ms`, and `inference_ms`:

```rust,no_run
# use yamabiko_asr::TranscriptSegment;
# fn emit_to_ui(segment: &TranscriptSegment) {
// app.emit("transcript-segment", segment)?;
# }
```

Multiple capture streams can share the same loaded ASR model. Register each
additional stream explicitly; closing one input flushes and releases only that
source. Segment timestamps use a shared session timeline and emitted segments
carry the allocated source identifier. `send_at` timestamps are rounded down
to a 16 kHz sample boundary; after anchoring the first chunk, plain `send`
advances continuously by sample count. Each segment also has a stable `SegmentId`;
consumers should upsert by that ID so later text or speaker revisions can
replace an earlier version:

```rust,no_run
use std::time::Instant;
use yamabiko_asr::{TranscriptEvent, Transcriber};

# async fn send_audio(
#     transcriber: Transcriber,
# ) -> yamabiko_asr::Result<()> {
let session_started = Instant::now();
let session = transcriber.start();
let system_audio = session.open_source().await?;
let (microphone, mut events, worker) = session.into_parts();

let producer = tokio::spawn(async move {
    microphone.send(vec![0.0; 1600]).await?;
    let system_started_at = session_started.elapsed();
    system_audio
        .send_at(system_started_at, vec![0.0; 1600])
        .await?;

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

## Examples

Transcribe a mono 16 kHz WAV file:

```powershell
cargo run --example audio_file -- .\models\parakeet-tdt_ctc-0.6b-ja-onnx .\audio.wav ja
```

The Windows-only `audio_input` example captures the default microphone and,
by default, the default output device through WASAPI loopback. Choose the mode
near the top of `examples/audio_input.rs` by commenting out one constant and
uncommenting the other:

```rust
const CAPTURE_SYSTEM_AUDIO: bool = true; // microphone + system audio
// const CAPTURE_SYSTEM_AUDIO: bool = false; // microphone only
```

Run the selected audio-input mode:

```powershell
cargo run --example audio_input -- .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

Press Ctrl+C to flush the final segments and stop. With system audio enabled,
output is labeled `[microphone]` or `[system]` using each segment's source
identifier. Enabled capture devices must expose an f32 default format.
Clock-drift correction between independently clocked capture devices is
planned for long-running sessions.

For the Japanese Parakeet TDT model, export ONNX files first:

```powershell
python tools/export_parakeet_tdt_ja.py
cargo run --example audio_input -- --vad-min-silence-ms 800 .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

For experimental multilingual Parakeet TDT v3 testing, omit the language
argument and use automatic language selection:

```powershell
python tools/export_parakeet_tdt_multilingual.py
cargo run --example audio_input -- --vad-min-silence-ms 800 .\models\parakeet-tdt-0.6b-v3-onnx
```

The local ONNX runner is used because `nvidia/parakeet-tdt_ctc-0.6b-ja`
expects 80 mel features.

The default build uses CPU execution. ONNX Runtime acceleration providers are
opt-in Cargo features, for example:

```powershell
cargo run --features directml --example audio_input -- --device directml .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
cargo run --features cuda --example audio_input -- --device cuda .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
cargo run --features openvino --example audio_input -- --device openvino .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

## License

This crate is licensed under either MIT or Apache-2.0, at your option. Model
files are not distributed by this crate; check each model's license before use.
