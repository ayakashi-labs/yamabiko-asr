# yamabiko-asr

On-device streaming speech transcription for desktop Rust applications, using
Silero VAD and local Parakeet TDT ONNX models.

> **Status:** This crate is pre-1.0 and its public API may change between minor
> releases. The current support target is Windows.

## Features

- Runs transcription locally; audio is not sent to a hosted service.
- Accepts streaming f32 mono 16 kHz PCM.
- Uses Silero VAD to remove silence and emit final utterance segments.
- Bounds retained audio during silence and splits continuous speech after 30
  seconds by default.
- Supports multiple independent audio sources with one shared ASR model and
  one shared Silero model session.
- Preserves source identifiers and timestamps on a shared session timeline.
- Supports CPU execution plus opt-in ONNX Runtime acceleration providers.
- Optionally serializes transcript events for Tauri or other UI layers.

## Requirements

- Rust 1.88 or newer.
- A Tokio runtime when starting a transcription session.
- A converted Parakeet TDT ONNX model directory containing:
  - `encoder.onnx` and, when exported separately, `encoder.onnx.data`
  - `decoder_joint.onnx`
  - `vocab.txt`
- Input audio converted to f32 mono 16 kHz PCM.

Audio capture, system-audio loopback, downmixing, resampling, and model download
belong to the application or integration layer. The repository examples show
one Windows implementation of that layer.

## Installation

```toml
[dependencies]
yamabiko-asr = "0.2"
```

Enable only the optional features your application uses. For example:

```toml
yamabiko-asr = { version = "0.2", features = ["serde", "directml"] }
```

## Model setup

The crate does not distribute model files. Clone this repository to use the
included conversion tools. They require a Python environment with PyTorch,
NVIDIA NeMo ASR, and ONNX.

The currently verified Japanese model is
`nvidia/parakeet-tdt_ctc-0.6b-ja`:

```powershell
python tools/export_parakeet_tdt_ja.py
```

The converted model is written to
`models/parakeet-tdt_ctc-0.6b-ja-onnx` by default.

Experimental multilingual conversion is also available:

```powershell
python tools/export_parakeet_tdt_multilingual.py
```

Run `nvidia/parakeet-tdt-0.6b-v3` with automatic language selection. Explicit
non-Japanese language hints are not accepted yet. Review the upstream model
license before distributing or using converted model files.

## Quick start

`AudioInput::send` accepts any chunk size. This example sends already-converted
PCM and then drains the final transcript events:

```rust
use yamabiko_asr::{Error, TranscriptEvent, Transcriber};

async fn transcribe(pcm: Vec<f32>) -> yamabiko_asr::Result<()> {
    let transcriber = Transcriber::builder("path/to/parakeet-tdt-model").build()?;
    let (input, mut events, worker) = transcriber.start().into_parts();

    input.send(pcm).await?;
    input.close().await?;

    while let Some(event) = events.recv().await {
        match event? {
            TranscriptEvent::Segment(segment) => {
                println!("{}ms: {}", segment.start_ms(), segment.text);
            }
            TranscriptEvent::EndOfStream => break,
            _ => {}
        }
    }

    worker.await.map_err(|_| Error::StreamClosed)?;
    Ok(())
}
```

Model loading and ONNX inference are synchronous. Build the `Transcriber` away
from a GUI thread. `Transcriber::start` moves inference onto Tokio's blocking
pool.

## Multiple audio sources

Additional sources share the loaded ASR and VAD models, but keep independent
PCM buffers, VAD state, and source-local timelines. Audio sources are not mixed
before transcription.

```rust
use std::time::Duration;
use yamabiko_asr::{Error, TranscriptEvent, Transcriber};

async fn transcribe_two_sources(
    microphone_pcm: Vec<f32>,
    system_pcm: Vec<f32>,
) -> yamabiko_asr::Result<()> {
    let transcriber = Transcriber::builder("path/to/parakeet-tdt-model").build()?;
    let session = transcriber.start();
    let system_audio = session.open_source().await?;
    let microphone_id = session.input.source_id();
    let system_id = system_audio.source_id();
    let (microphone, mut events, worker) = session.into_parts();

    microphone
        .send_at(Duration::ZERO, microphone_pcm)
        .await?;
    system_audio.send_at(Duration::ZERO, system_pcm).await?;

    system_audio.close().await?;
    microphone.close().await?;

    while let Some(event) = events.recv().await {
        match event? {
            TranscriptEvent::Segment(segment) => {
                let source = if segment.source_id == microphone_id {
                    "microphone"
                } else if segment.source_id == system_id {
                    "system"
                } else {
                    "unknown"
                };
                println!("[{source}] {}", segment.text);
            }
            TranscriptEvent::EndOfStream => break,
            _ => {}
        }
    }

    worker.await.map_err(|_| Error::StreamClosed)?;
    Ok(())
}
```

The first `send_at` anchors a source to the shared session timeline. Following
chunks can use `send`; they continue from the preceding sample count. Capture
integrations should derive the first timestamp from the audio device clock.

`max_sources` defaults to 2, including the primary input:

```rust
use yamabiko_asr::Transcriber;

fn configured_transcriber() -> yamabiko_asr::Result<Transcriber> {
    Transcriber::builder("path/to/model")
        .max_sources(4)
        .build()
}
```

## Transcript events and serialization

Each `TranscriptSegment` contains a stable `SegmentId`, `AudioSourceId`, text,
session-relative start and end times, inference duration, finality, and an
optional speaker identifier.

With the `serde` feature enabled, `TranscriptEvent` and `TranscriptSegment`
implement `Serialize` directly. Duration fields are serialized as `start_ms`,
`end_ms`, and `inference_ms`, so a separate UI payload type is unnecessary.

## Cargo features

| Feature | Purpose |
| --- | --- |
| `serde` | Implements `Serialize` for transcript events, segments, and identifiers. |
| `directml` | Enables the ONNX Runtime DirectML execution provider. |
| `cuda` | Enables the CUDA execution provider. |
| `tensorrt` | Enables the TensorRT execution provider. |
| `openvino` | Enables the OpenVINO execution provider. |
| `rocm` | Enables the ROCm execution provider. |
| `coreml` | Enables the Core ML execution provider. |
| `xnnpack` | Enables the XNNPACK execution provider. |
| `onednn` | Enables the oneDNN execution provider. |

The default build uses CPU execution. Selecting an explicit accelerator with
`TranscriberBuilder::device` requires the matching Cargo feature and its
runtime libraries. `Device::Auto` tries available providers before CPU.

## Repository examples

Run these commands from a clone of this repository.

### WAV file

`audio_file` streams a mono 16 kHz WAV instead of loading the entire file:

```powershell
cargo run --example audio_file -- .\models\parakeet-tdt_ctc-0.6b-ja-onnx .\audio.wav ja
```

### Microphone and system audio

The Windows-only `audio_input` example captures the default microphone and,
by default, the default output device through WASAPI loopback. It downmixes and
resamples both sources before sending them to the crate.

Choose the capture mode near the top of `examples/audio_input.rs`:

```rust
const CAPTURE_SYSTEM_AUDIO: bool = true; // microphone + system audio
// const CAPTURE_SYSTEM_AUDIO: bool = false; // microphone only
```

Run the selected mode:

```powershell
cargo run --example audio_input -- .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

Press Ctrl+C to flush the final segments and stop. Captured devices must expose
an f32 default format. To use an accelerator, enable its feature and select the
same device:

```powershell
cargo run --features directml --example audio_input -- --device directml .\models\parakeet-tdt_ctc-0.6b-ja-onnx ja
```

The examples also accept `--vad-threshold`, `--vad-min-speech-ms`,
`--vad-min-silence-ms`, and `--vad-speech-pad-ms`.

## Known limitations and roadmap

- The current supported platform is Windows.
- The public PCM boundary is fixed to f32 mono 16 kHz.
- The Japanese Parakeet TDT-CTC model is the currently verified model path;
  multilingual support remains experimental.
- Events contain VAD-final utterances; partial transcript updates are not
  emitted yet.
- Speaker diarization and speaker identification are planned but not
  implemented.
- Long-running microphone/system clock-drift correction is planned but not
  implemented.
- Multiple sources share one ASR model and are processed sequentially rather
  than inferred in parallel.
- Transcript events use an unbounded channel. Long-running applications should
  drain events continuously.

## License

This crate is licensed under either MIT or Apache-2.0, at your option. Model
files are not distributed by this crate; check each model's license before use.
