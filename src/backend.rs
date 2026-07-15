use crate::Result;

/// A loaded, heavyweight ASR model shared by recognition streams.
pub(crate) trait AsrModel: Send {
    fn transcribe(&mut self, samples: Vec<f32>) -> Result<String>;
}

impl AsrModel for crate::tdt::ParakeetTdtModel {
    fn transcribe(&mut self, samples: Vec<f32>) -> Result<String> {
        self.transcribe_samples(samples)
    }
}
