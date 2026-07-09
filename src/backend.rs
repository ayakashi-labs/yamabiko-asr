use crate::tdt::JapaneseTdtModel;
use crate::vad::SpeechChunk;
use crate::{Device, Error, Language, Result, TranscriberConfig};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron, NemotronMode};

pub(crate) trait StreamingAsrBackend: Send {
    fn wants_partial_speech(&self) -> bool;
    fn accept_speech(
        &mut self,
        speech: &SpeechChunk,
        language: &Language,
    ) -> Result<Vec<BackendTranscript>>;
    fn flush(&mut self, next_input_sample: u64) -> Result<Vec<BackendTranscript>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTranscript {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub is_final: bool,
}

pub(crate) struct NemotronBackend {
    model: Nemotron,
    last_segment: Option<BackendTranscript>,
}

impl NemotronBackend {
    pub(crate) fn load(config: &TranscriberConfig) -> Result<Self> {
        let exec_config = execution_config(config.device)?;
        let mut model = Nemotron::from_pretrained(&config.model_dir, Some(exec_config))
            .map_err(|err| Error::ModelLoad(err.to_string()))?;

        if model.mode() == NemotronMode::Multilingual {
            if let Some(language) = config.language.as_backend_code() {
                model
                    .set_target_lang(&language)
                    .map_err(|err| Error::InvalidLanguageHint(err.to_string()))?;
            }
        } else if !matches!(config.language, Language::Auto) {
            return Err(Error::InvalidLanguageHint(
                "language hints require a multilingual Nemotron model".to_string(),
            ));
        }

        Ok(Self {
            model,
            last_segment: None,
        })
    }
}

impl StreamingAsrBackend for NemotronBackend {
    fn wants_partial_speech(&self) -> bool {
        true
    }

    fn accept_speech(
        &mut self,
        speech: &SpeechChunk,
        _language: &Language,
    ) -> Result<Vec<BackendTranscript>> {
        if speech.samples.is_empty() {
            if speech.is_final
                && let Some(mut segment) = self.last_segment.take()
            {
                segment.end_sample = segment.end_sample.max(speech.end_sample);
                segment.is_final = true;
                return Ok(vec![segment]);
            }
            return Ok(Vec::new());
        }

        let text = self
            .model
            .transcribe_chunk(&speech.samples)
            .map_err(|err| Error::Backend(err.to_string()))?;

        let transcript = BackendTranscript {
            text,
            start_sample: speech.start_sample,
            end_sample: speech.end_sample,
            is_final: speech.is_final,
        };

        if transcript.is_final {
            self.last_segment = None;
        } else if !transcript.text.trim().is_empty() {
            self.last_segment = Some(transcript.clone());
        }

        Ok(vec![transcript])
    }

    fn flush(&mut self, _next_input_sample: u64) -> Result<Vec<BackendTranscript>> {
        Ok(Vec::new())
    }
}

pub(crate) struct ParakeetTdtBackend {
    model: JapaneseTdtModel,
    pending_samples: Vec<f32>,
    pending_start_sample: Option<u64>,
    pending_end_sample: u64,
}

impl ParakeetTdtBackend {
    pub(crate) fn load(config: &TranscriberConfig) -> Result<Self> {
        validate_tdt_language(&config.language)?;
        let model = JapaneseTdtModel::load(&config.model_dir, config.device)?;

        Ok(Self {
            model,
            pending_samples: Vec::new(),
            pending_start_sample: None,
            pending_end_sample: 0,
        })
    }

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

    fn transcribe_pending(&mut self, default_end_sample: u64) -> Result<Vec<BackendTranscript>> {
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

        let result = self
            .model
            .transcribe_samples(samples)
            .map_err(|err| Error::Backend(err.to_string()))?;

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
}

impl StreamingAsrBackend for ParakeetTdtBackend {
    fn wants_partial_speech(&self) -> bool {
        false
    }

    fn accept_speech(
        &mut self,
        speech: &SpeechChunk,
        _language: &Language,
    ) -> Result<Vec<BackendTranscript>> {
        self.push_samples(speech);
        if speech.is_final {
            self.transcribe_pending(speech.end_sample)
        } else {
            Ok(Vec::new())
        }
    }

    fn flush(&mut self, next_input_sample: u64) -> Result<Vec<BackendTranscript>> {
        self.transcribe_pending(next_input_sample)
    }
}

fn execution_config(device: Device) -> Result<ExecutionConfig> {
    let provider = match device {
        Device::Cpu => ExecutionProvider::Cpu,
        Device::DirectMl => ExecutionProvider::DirectML,
    };

    Ok(ExecutionConfig::new().with_execution_provider(provider))
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
