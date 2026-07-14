# Changelog

All notable changes to this project are documented in this file.

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

[0.2.1]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ayakashi-labs/yamabiko-asr/compare/v0.1.0...v0.2.0
