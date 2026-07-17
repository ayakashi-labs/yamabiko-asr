# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.3.2] - 2026-07-17

### Added

- Implemented `futures_core::Stream` for `TranscriptEventReceiver`, enabling
  composition with compatible `StreamExt` utilities while preserving output
  metrics and cancellation behavior.

### Changed

- Clarified that `send`, `send_at`, and their blocking variants report input
  queue acceptance rather than VAD or transcription completion, with later
  processing and timestamp errors delivered through session events.

### Fixed

- Ensured dropping an audio input always queues an ordered source close even
  when audio capacity is exhausted, so buffered speech is flushed and source
  slots are released.
- Preserved configured `input_capacity` backpressure for audio while allowing
  close and other control commands to bypass audio capacity.

## [0.3.1] - 2026-07-16

### Fixed

- Matched local Parakeet TDT feature extraction to NeMo framing and
  normalization, including centered STFT windows, valid-frame masking, and
  encoder-length handling.
- Validated ONNX graph inputs and outputs, vocabulary IDs, decoder logits, and
  recurrent state tensors before consuming model data.
- Hardened receiver cancellation and source-close races while preserving
  lossless draining and the documented terminal event order.
- Bounded the Windows live-capture PCM handoff and preserved capture,
  forwarding, input-close, and worker failure causes during shutdown.
- Rejected non-zero VAD durations that truncate to zero samples.

### Changed

- Added README Rust code blocks to the documentation-test CI gate.
- Removed redundant internal model and pipeline wrapper state.

## [0.3.0] - 2026-07-15

### Breaking changes

- Removed the unused `Language` type, `Error::InvalidLanguageHint`, and the
  `TranscriberBuilder::language` and `language_hint` methods. Language behavior
  is now determined entirely by the loaded model.
- Replaced the raw Tokio event receiver exposed by `TranscriptionSession` and
  `into_parts()` with `TranscriptEventReceiver`.

### Added

- Added `OutputMonitor` and `OutputMetrics` snapshots for pending, peak,
  emitted, received, discarded, and failed transcript-event delivery counts.
- Added output-backlog warnings and a shutdown metrics summary to the
  `audio_input` example.
- Added Windows CI coverage for formatting, Clippy, tests, packaging, and the
  Rust 1.88 minimum supported version.

### Changed

- Kept transcript delivery unbounded and lossless while exposing backlog
  visibility for long-running applications.
- Made closing or dropping the transcript receiver cancel subsequent input
  processing and worker output without discarding events that remain available
  for draining after an explicit close.
- Documented the 0.2 migration, output monitoring, receiver cancellation, and
  the intentional absence of partial transcript updates.

## [0.2.1] - 2026-07-14

### Fixed

- Replaced Rustdoc-only hidden scaffolding in README snippets with complete
  Rust functions so examples render clearly on GitHub and crates.io.
- Compiled every Rust code block in the README as part of release validation.

## [0.2.0] - 2026-07-14

### Breaking changes

- Replaced `PcmChunk` and `PcmFormat` inputs with fixed-format
  `Vec<f32>` input methods. Callers must provide mono 16 kHz PCM.
- Made `TranscriberConfig` and `VadConfig` internal and standardized
  construction on `Transcriber::builder`.
- Renamed `channel_capacity` to `input_capacity`.
- Simplified additional source registration to `open_source()` and removed
  source configuration and source-kind types.
- Removed transcript payload mirror types and `to_payload()`. With the `serde`
  feature, transcript events and segments now serialize directly.
- Removed `TranscriptionSession::into_channels`; use `into_parts` to retain the
  worker handle.

### Added

- Added multiple independent audio sources sharing one loaded ASR model.
- Added shared-session timeline anchoring with `send_at` and source identifiers
  on transcript segments.
- Added configurable maximum VAD utterance duration, defaulting to 30 seconds.
- Added Windows microphone and WASAPI system-audio capture to the
  `audio_input` example.
- Added streaming WAV input and integer PCM normalization tests.

### Changed

- Shared one Silero model session across audio sources while preserving
  independent stream and segmentation state.
- Made transcript delivery unbounded so closing an input does not depend on
  concurrently draining events.
- Reduced TDT preprocessing, encoder, and decoder allocations and copies.
- Streamed WAV examples in bounded chunks instead of loading complete files.
- Reduced default and development dependency features.

### Fixed

- Bounded retained VAD PCM during long silence and continuous speech.
- Preserved the final partial-frame utterance when closing an input.
- Corrected 8-, 16-, 24-, and 32-bit integer WAV normalization.
- Flushed and released each audio source independently.

[Unreleased]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.1.0...v0.2.0
