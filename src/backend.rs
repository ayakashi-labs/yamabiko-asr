use crate::tdt::ParakeetTdtModel as LoadedParakeetTdtModel;
use crate::{Error, Result, TranscriberConfig};

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
