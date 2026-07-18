use yamabiko_asr::{
    AudioSourceOptions, Device, DiarizationConfig, DiarizationMode, SpeakerActivity, Transcriber,
    TranscriptEvent, TranscriptionSession,
};

#[test]
fn public_diarization_configuration_is_constructible() {
    let _builder = Transcriber::builder("asr-model")
        .device(Device::DirectMl)
        .diarization(DiarizationConfig::new("diarization-model"));
    let options = AudioSourceOptions::new().diarization(DiarizationMode::On);

    assert_ne!(options, AudioSourceOptions::default());

    let _start: fn(Transcriber, AudioSourceOptions) -> yamabiko_asr::Result<TranscriptionSession> =
        Transcriber::start_with_options;
}

#[allow(dead_code)]
async fn open_source_with_public_options(
    session: &TranscriptionSession,
    options: AudioSourceOptions,
) {
    let _ = session.open_source_with_options(options).await;
}

#[test]
fn speaker_activity_is_exposed_as_a_transcript_event() {
    let _constructor: fn(SpeakerActivity) -> TranscriptEvent = TranscriptEvent::SpeakerActivity;
}
