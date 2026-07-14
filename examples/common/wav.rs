use crate::common::ExampleResult;
use hound::{SampleFormat, WavReader};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const SAMPLE_RATE: u32 = yamabiko_asr::PCM_SAMPLE_RATE_HZ;

pub struct WavPcmReader {
    reader: WavReader<BufReader<File>>,
    encoding: Encoding,
}

#[derive(Clone, Copy)]
enum Encoding {
    Float,
    Int8 { scale: f32 },
    Int16 { scale: f32 },
    Int32 { scale: f32 },
}

impl WavPcmReader {
    pub fn open(path: impl AsRef<Path>) -> ExampleResult<Self> {
        let reader = WavReader::open(path)?;
        let spec = reader.spec();
        if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
            return Err(
                "expected mono 16 kHz WAV; resample/downmix before using this crate".into(),
            );
        }

        let encoding = match spec.sample_format {
            SampleFormat::Float => Encoding::Float,
            SampleFormat::Int if spec.bits_per_sample <= 8 => Encoding::Int8 {
                scale: integer_scale(spec.bits_per_sample)?,
            },
            SampleFormat::Int if spec.bits_per_sample <= 16 => Encoding::Int16 {
                scale: integer_scale(spec.bits_per_sample)?,
            },
            SampleFormat::Int => Encoding::Int32 {
                scale: integer_scale(spec.bits_per_sample)?,
            },
        };

        Ok(Self { reader, encoding })
    }

    pub fn read_chunk(&mut self, max_samples: usize) -> ExampleResult<Option<Vec<f32>>> {
        let mut chunk = Vec::with_capacity(max_samples);
        while chunk.len() < max_samples {
            let sample = match self.encoding {
                Encoding::Float => self.reader.samples::<f32>().next().transpose()?,
                Encoding::Int8 { scale } => self
                    .reader
                    .samples::<i8>()
                    .next()
                    .transpose()?
                    .map(|sample| sample as f32 / scale),
                Encoding::Int16 { scale } => self
                    .reader
                    .samples::<i16>()
                    .next()
                    .transpose()?
                    .map(|sample| sample as f32 / scale),
                Encoding::Int32 { scale } => self
                    .reader
                    .samples::<i32>()
                    .next()
                    .transpose()?
                    .map(|sample| sample as f32 / scale),
            };

            match sample {
                Some(sample) => chunk.push(sample),
                None => break,
            }
        }

        Ok((!chunk.is_empty()).then_some(chunk))
    }
}

fn integer_scale(bits_per_sample: u16) -> ExampleResult<f32> {
    if !(1..=32).contains(&bits_per_sample) {
        return Err(format!("unsupported integer WAV bit depth: {bits_per_sample}").into());
    }
    Ok((1u64 << (bits_per_sample - 1)) as f32)
}
