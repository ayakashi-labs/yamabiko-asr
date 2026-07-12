use crate::tdt::ParakeetTdtModel as LoadedParakeetTdtModel;
use crate::vad::SpeechChunk;
use crate::{Error, Language, Result, TranscriberConfig};

/// A loaded, heavyweight ASR model shared by recognition streams.
pub(crate) trait AsrModel: Send {
    fn transcribe(&mut self, samples: Vec<f32>, language: &Language) -> Result<String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTranscript {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub is_final: bool,
}

/// The loaded Parakeet model. This owns the expensive ONNX sessions.
pub(crate) struct ParakeetTdtModel {
    inner: LoadedParakeetTdtModel,
}

impl ParakeetTdtModel {
    pub(crate) fn load(config: &TranscriberConfig) -> Result<Self> {
        validate_tdt_language(&config.language)?;
        Ok(Self {
            inner: LoadedParakeetTdtModel::load(&config.model_dir, config.device)?,
        })
    }
}

impl AsrModel for ParakeetTdtModel {
    fn transcribe(&mut self, samples: Vec<f32>, _language: &Language) -> Result<String> {
        self.inner
            .transcribe_samples(samples)
            .map_err(|err| Error::Backend(err.to_string()))
    }
}

/// Lightweight buffering and decoder input state for one audio source.
#[derive(Default)]
pub(crate) struct RecognitionStream {
    pending_samples: Vec<f32>,
    pending_start_sample: Option<u64>,
    pending_end_sample: u64,
}

impl RecognitionStream {
    fn push_samples(&mut self, speech: &SpeechChunk) {
        if !speech.samples.is_empty() {
            if self.pending_start_sample.is_none() {
                self.pending_start_sample = Some(speech.start_sample);
            }
            self.pending_samples.extend_from_slice(&speech.samples);
        }

        if self.pending_start_sample.is_some() {
            self.pending_end_sample = self.pending_end_sample.max(speech.end_sample);
        }
    }

    fn transcribe_pending(
        &mut self,
        model: &mut dyn AsrModel,
        language: &Language,
        default_end_sample: u64,
    ) -> Result<Vec<BackendTranscript>> {
        let Some(start_sample) = self.pending_start_sample.take() else {
            return Ok(Vec::new());
        };
        if self.pending_samples.is_empty() {
            self.pending_end_sample = 0;
            return Ok(Vec::new());
        }

        let end_sample = self.pending_end_sample.max(default_end_sample);
        let samples = std::mem::take(&mut self.pending_samples);
        self.pending_end_sample = 0;
        let result = model.transcribe(samples, language)?;

        if result.trim().is_empty() {
            return Ok(Vec::new());
        }

        Ok(vec![BackendTranscript {
            text: result,
            start_sample,
            end_sample,
            is_final: true,
        }])
    }

    pub(crate) fn accept_speech(
        &mut self,
        model: &mut dyn AsrModel,
        speech: &SpeechChunk,
        language: &Language,
    ) -> Result<Vec<BackendTranscript>> {
        self.push_samples(speech);
        if speech.is_final {
            self.transcribe_pending(model, language, speech.end_sample)
        } else {
            Ok(Vec::new())
        }
    }

    pub(crate) fn flush(
        &mut self,
        model: &mut dyn AsrModel,
        language: &Language,
        next_input_sample: u64,
    ) -> Result<Vec<BackendTranscript>> {
        self.transcribe_pending(model, language, next_input_sample)
    }
}

fn validate_tdt_language(language: &Language) -> Result<()> {
    match language {
        Language::Auto => Ok(()),
        Language::Hint(hint)
            if hint.eq_ignore_ascii_case("ja") || hint.eq_ignore_ascii_case("ja-JP") =>
        {
            Ok(())
        }
        Language::Hint(hint) => Err(Error::InvalidLanguageHint(format!(
            "ParakeetTDT backend currently accepts auto or ja/ja-JP for the Japanese model, got {hint}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdt_language_validation_accepts_japanese_or_auto() {
        assert!(validate_tdt_language(&Language::Auto).is_ok());
        assert!(validate_tdt_language(&Language::Hint("ja".to_string())).is_ok());
        assert!(validate_tdt_language(&Language::Hint("ja-JP".to_string())).is_ok());
    }

    #[test]
    fn tdt_language_validation_rejects_non_japanese_hints() {
        let err = validate_tdt_language(&Language::Hint("en-US".to_string())).unwrap_err();
        assert!(matches!(err, Error::InvalidLanguageHint(_)));
    }
}
