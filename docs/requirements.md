# Parakeet Transcription Requirements

## Purpose

This crate provides an easy Parakeet-family transcription layer for desktop
apps, especially Tauri 2.0 apps on Windows. It uses Silero VAD for speech
gating and a small local ONNX runner for Parakeet TDT models that need 80 mel
features. Users provide the PCM input pipeline and compatible local model files.

## v0.1 Scope

- Accept only `f32`, mono, 16 kHz PCM.
- Provide a Tokio-based async streaming input/output API.
- Run exactly one Parakeet transcription engine per `Transcriber`.
- Emit VAD-final transcript segment events with input-audio timestamps.
- Gate audio with Silero VAD before sending speech to the ASR backend.
- Expose VAD threshold, minimum speech duration, minimum silence duration, and
  speech padding.
- Support automatic language selection plus explicit `ja`/`ja-JP` hints for the
  current Japanese Parakeet TDT path.
- Let callers explicitly select the Parakeet TDT ONNX execution device. v0.1
  supports `cpu`, `auto`, `directml`, `cuda`, `tensorrt`, `openvino`, `rocm`,
  `coreml`, `xnnpack`, and `onednn` in the public API. Non-CPU providers
  require matching Cargo features and runtime libraries. The default build and
  default `Device` use CPU. `auto` may fall back to CPU; explicit accelerator
  choices should surface registration failures as device errors. WebGPU is not
  exposed in v0.1 because the current ORT package does not link reliably with
  all enabled features on Windows.
- Support `nvidia/parakeet-tdt_ctc-0.6b-ja` as the first model target. The
  current backend emits only final utterance results after VAD closes a speech
  segment.
- Keep Parakeet TDT semantics explicit: VAD receives streaming PCM, but the ASR
  backend receives the finalized VAD speech segment to avoid losing early
  speech.

## Out of Scope for v0.1

- Audio capture, system audio capture, resampling, and downmixing.
- Runtime model downloads or model license handling. Repository tools may offer
  manual export/conversion helpers for local testing.
- Non-Parakeet ASR models such as Nemotron.
- Partial transcript output and true ASR backend streaming.
- Speaker diarization, multi-PCM input, translation, and TUI.

## Future Work

- Speaker diarization should be added as an optional feature using Sortformer.
- Multi-PCM support should merge inputs before the single ASR engine.
- Parakeet multilingual models such as `nvidia/parakeet-tdt-0.6b-v3` should be
  supported through a separate export script. Runtime behavior and language
  validation should be verified per model before documenting production use.
- True streaming partial output can be added when a Parakeet backend supports
  it reliably in this crate.
- Translation can be added after transcription semantics are stable.
- The initial public crate name is `asr-crate`.
- The crate license is `MIT OR Apache-2.0`. Model files are not distributed by
  this crate and remain governed by their upstream licenses.

## Quality Requirements

- Public APIs should hide backend crate types where possible.
- Public errors must be actionable and matchable.
- Tests should cover VAD gating, timestamp continuity, stream event order,
  configuration validation, and Parakeet TDT VAD chunking behavior.
- Examples should include WAV pseudo-streaming, Windows-oriented microphone
  input, VAD tuning flags, device selection flags, and documented model
  conversion commands.
