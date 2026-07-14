use crate::tdt::ParakeetTdtModel as LoadedParakeetTdtModel;
use crate::{Error, Language, Result, TranscriberConfig};

/// A loaded, heavyweight ASR model shared by recognition streams.
pub(crate) trait AsrModel: Send {
    fn transcribe(&mut self, samples: Vec<f32>) -> Result<String>;
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
    fn transcribe(&mut self, samples: Vec<f32>) -> Result<String> {
        self.inner
            .transcribe_samples(samples)
            .map_err(|err| Error::Backend(err.to_string()))
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
