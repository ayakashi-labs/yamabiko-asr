# Multilingual Streaming Transcription Requirements

## Purpose

This crate provides an easy streaming transcription layer for desktop apps,
especially Tauri 2.0 apps on Windows. It uses Silero VAD for speech gating,
`parakeet-rs` for Nemotron streaming ASR, and a small local ONNX runner for
Japanese Parakeet TDT models that need 80 mel features. Users provide the PCM
input pipeline and compatible local model files.

## v0.1 Scope

- Accept only `f32`, mono, 16 kHz PCM.
- Provide a Tokio-based async streaming API.
- Run exactly one transcription engine per `Transcriber`.
- Emit partial/final transcript segment events with input-audio timestamps.
- Gate audio with Silero VAD before sending speech to the ASR backend.
- Expose VAD threshold, minimum speech duration, minimum silence duration, and
  speech padding.
- Support automatic language selection plus explicit hints such as `ja` and
  `en` where the selected backend supports them.
- Let callers explicitly select CPU or DirectML execution. Provider fallback
  follows the behavior of the underlying `parakeet-rs`/ONNX Runtime backend.
- Provide backend selection. `Nemotron` is the true streaming default.
  `ParakeetTDT` supports Japanese TDT experiments with
  `nvidia/parakeet-tdt_ctc-0.6b-ja`, but emits only final utterance results
  after VAD closes a speech segment. The Japanese TDT path uses an 80 mel ONNX
  runner because the current `parakeet-rs::ParakeetTDT` helper assumes 128 mel
  input.
- Keep backend-specific streaming semantics explicit: Nemotron receives active
  speech chunks for partial output, while Parakeet TDT receives the finalized VAD
  speech segment to avoid losing early speech.
- Limit Parakeet TDT language hints to `auto`, `ja`, and `ja-JP`.

## Out of Scope for v0.1

- Audio capture, system audio capture, resampling, and downmixing.
- Runtime model downloads or model license handling. Repository tools may offer
  manual export/conversion helpers for local testing.
- Speaker diarization, multi-PCM input, translation, and TUI.

## Future Work

- Speaker diarization should be added as an optional feature using Sortformer.
- Multi-PCM support should merge inputs before the single ASR engine.
- Translation can be added after transcription semantics are stable.
- The public crate name and license must be decided before publication.

## Quality Requirements

- Public APIs should hide backend crate types where possible.
- Public errors must be actionable and matchable.
- Tests should cover VAD gating, timestamp continuity, stream event order,
  configuration validation, and backend-specific VAD chunking behavior.
- Examples should include WAV pseudo-streaming, Windows-oriented microphone
  input, VAD tuning flags, and documented model conversion commands.
