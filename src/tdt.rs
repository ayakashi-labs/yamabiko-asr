use crate::{Device, Error, PCM_SAMPLE_RATE_HZ, Result};
use ndarray::{Array1, Array2, Array3};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::ValueType;
use realfft::RealToComplex;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;
const SENTENCEPIECE_SPACE: char = '\u{2581}';

pub(crate) struct ParakeetTdtModel {
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    word_boundary: WordBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordBoundary {
    Strip,
    Space,
}

impl ParakeetTdtModel {
    pub(crate) fn load(model_dir: &Path, device: Device) -> Result<Self> {
        if !model_dir.is_dir() {
            return Err(Error::ModelLoad(format!(
                "TDT model path must be a directory: {}",
                model_dir.display()
            )));
        }

        let encoder = build_session(&find_encoder(model_dir)?, device)?;
        let decoder_joint = build_session(&find_decoder_joint(model_dir)?, device)?;
        let feature_size = encoder_feature_size(&encoder)?;
        let vocab = load_vocab(&model_dir.join("vocab.txt"))?;
        let word_boundary = word_boundary_for_vocab(&vocab);
        let mel_basis = create_mel_filterbank(N_FFT, feature_size, PCM_SAMPLE_RATE_HZ as usize);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
            mel_basis,
            fft_plan,
            word_boundary,
        })
    }

    pub(crate) fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<String> {
        let features = self.extract_features(samples)?;
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;
        let tokens = self.greedy_decode(&encoder_out, encoder_len)?;
        Ok(self.decode_tokens(&tokens))
    }

    fn extract_features(&self, mut audio: Vec<f32>) -> Result<Array2<f32>> {
        audio = apply_preemphasis(&audio, PREEMPHASIS);
        let spectrogram = stft_with_plan(&audio, &self.fft_plan, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel_spectrogram = self.mel_basis.dot(&spectrogram);
        let log_zero_guard = 2.0f32.powi(-24);
        let mel_spectrogram = mel_spectrogram.mapv(|value| (value + log_zero_guard).ln());
        let mut mel_spectrogram = mel_spectrogram.t().to_owned();

        let num_frames = mel_spectrogram.shape()[0];
        let num_features = mel_spectrogram.shape()[1];
        if num_frames <= 1 {
            return Ok(mel_spectrogram);
        }

        for feat_idx in 0..num_features {
            let mut column = mel_spectrogram.column_mut(feat_idx);
            let mean = column.iter().sum::<f32>() / num_frames as f32;
            let variance = column
                .iter()
                .map(|&value| (value - mean).powi(2))
                .sum::<f32>()
                / (num_frames as f32 - 1.0);
            let std = variance.sqrt() + 1e-5;
            for value in column.iter_mut() {
                *value = (*value - mean) / std;
            }
        }

        Ok(mel_spectrogram)
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];
        let input = features
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(|err| Error::Backend(format!("failed to reshape encoder input: {err}")))?
            .to_owned();
        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let outputs = self
            .encoder
            .run(ort::inputs!(
                "audio_signal" => ort::value::Value::from_array(input)
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "length" => ort::value::Value::from_array(input_length)
                    .map_err(|err| Error::Backend(err.to_string()))?
            ))
            .map_err(|err| Error::Backend(err.to_string()))?;

        let encoder_out = &outputs["outputs"];
        let encoder_lens = &outputs["encoded_lengths"];
        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|err| Error::Backend(format!("failed to extract encoder output: {err}")))?;
        let (_, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|err| Error::Backend(format!("failed to extract encoder length: {err}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Backend(format!(
                "expected 3D encoder output, got shape {shape_dims:?}"
            )));
        }

        let encoder_array = Array3::from_shape_vec(
            (
                shape_dims[0] as usize,
                shape_dims[1] as usize,
                shape_dims[2] as usize,
            ),
            data.to_vec(),
        )
        .map_err(|err| Error::Backend(format!("failed to create encoder array: {err}")))?;

        Ok((encoder_array, lens_data[0]))
    }

    fn greedy_decode(&mut self, encoder_out: &Array3<f32>, encoder_len: i64) -> Result<Vec<usize>> {
        let encoder_dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2].min(encoder_len.max(0) as usize);
        let vocab_size = self.vocab.len();
        let blank_id = vocab_size.saturating_sub(1);
        let max_tokens_per_step = 10;

        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));
        let mut tokens = Vec::new();
        let mut emitted_tokens = 0;
        let mut last_emitted_token = blank_id as i32;
        let mut t = 0;

        while t < time_steps {
            let frame = encoder_out.slice(ndarray::s![0, .., t]).to_owned();
            let frame_reshaped = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|err| Error::Backend(format!("failed to reshape decoder frame: {err}")))?
                .to_owned();
            let targets = Array2::from_shape_vec((1, 1), vec![last_emitted_token])
                .map_err(|err| Error::Backend(format!("failed to create decoder target: {err}")))?;

            let outputs = self
                .decoder_joint
                .run(ort::inputs!(
                    "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)
                        .map_err(|err| Error::Backend(err.to_string()))?,
                    "targets" => ort::value::Value::from_array(targets)
                        .map_err(|err| Error::Backend(err.to_string()))?,
                    "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))
                        .map_err(|err| Error::Backend(err.to_string()))?,
                    "input_states_1" => ort::value::Value::from_array(state_h.clone())
                        .map_err(|err| Error::Backend(err.to_string()))?,
                    "input_states_2" => ort::value::Value::from_array(state_c.clone())
                        .map_err(|err| Error::Backend(err.to_string()))?
                ))
                .map_err(|err| Error::Backend(err.to_string()))?;

            let (_, logits_data) =
                outputs["outputs"]
                    .try_extract_tensor::<f32>()
                    .map_err(|err| {
                        Error::Backend(format!("failed to extract decoder logits: {err}"))
                    })?;
            let token_id = logits_data
                .iter()
                .take(vocab_size)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);
            let duration_step = logits_data
                .iter()
                .skip(vocab_size)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if token_id != blank_id {
                state_h = extract_state(&outputs["output_states_1"], "output_states_1")?;
                state_c = extract_state(&outputs["output_states_2"], "output_states_2")?;
                tokens.push(token_id);
                last_emitted_token = token_id as i32;
                emitted_tokens += 1;
            }

            if duration_step > 0 {
                t += duration_step;
                emitted_tokens = 0;
            } else if token_id == blank_id || emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok(tokens)
    }

    fn decode_tokens(&self, tokens: &[usize]) -> String {
        let mut text = String::new();
        for token_id in tokens {
            let Some(token_text) = self.vocab.get(*token_id) else {
                continue;
            };
            if token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>" {
                continue;
            }
            let replacement = match self.word_boundary {
                WordBoundary::Strip => "",
                WordBoundary::Space => " ",
            };
            text.push_str(&token_text.replace(SENTENCEPIECE_SPACE, replacement));
        }
        text.trim().to_string()
    }
}

fn word_boundary_for_vocab(vocab: &[String]) -> WordBoundary {
    if vocab.iter().any(|token| token == "<|en|>") {
        WordBoundary::Space
    } else {
        WordBoundary::Strip
    }
}

fn build_session(model_path: &Path, device: Device) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|err| Error::ModelLoad(err.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| Error::ModelLoad(err.to_string()))?
        .with_intra_threads(4)
        .map_err(|err| Error::ModelLoad(err.to_string()))?
        .with_inter_threads(1)
        .map_err(|err| Error::ModelLoad(err.to_string()))?;

    builder = builder
        .with_execution_providers(execution_providers_for(device))
        .map_err(|err| Error::DeviceUnavailable {
            device,
            message: err.to_string(),
        })?;

    builder
        .commit_from_file(model_path)
        .map_err(|err| Error::ModelLoad(err.to_string()))
}

fn execution_providers_for(device: Device) -> Vec<ExecutionProviderDispatch> {
    match device {
        Device::Cpu => vec![cpu_provider()],
        Device::Auto => vec![
            ort::ep::DirectML::default().build(),
            ort::ep::CUDA::default().build(),
            ort::ep::TensorRT::default().build(),
            ort::ep::OpenVINO::default().build(),
            ort::ep::ROCm::default().build(),
            ort::ep::CoreML::default().build(),
            ort::ep::XNNPACK::default().build(),
            ort::ep::OneDNN::default().build(),
            cpu_provider(),
        ],
        Device::DirectMl => vec![
            ort::ep::DirectML::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::Cuda => vec![
            ort::ep::CUDA::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::TensorRt => vec![
            ort::ep::TensorRT::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::OpenVino => vec![
            ort::ep::OpenVINO::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::Rocm => vec![
            ort::ep::ROCm::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::CoreMl => vec![
            ort::ep::CoreML::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::Xnnpack => vec![
            ort::ep::XNNPACK::default().build().error_on_failure(),
            cpu_provider(),
        ],
        Device::OneDnn => vec![
            ort::ep::OneDNN::default().build().error_on_failure(),
            cpu_provider(),
        ],
    }
}

fn cpu_provider() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().build().error_on_failure()
}

fn encoder_feature_size(encoder: &Session) -> Result<usize> {
    let Some(input) = encoder
        .inputs()
        .iter()
        .find(|input| input.name() == "audio_signal")
    else {
        return Err(Error::ModelLoad(
            "encoder model is missing audio_signal input".to_string(),
        ));
    };

    let ValueType::Tensor { shape, .. } = input.dtype() else {
        return Err(Error::ModelLoad(format!(
            "encoder audio_signal input must be a tensor, got {:?}",
            input.dtype()
        )));
    };

    let Some(&feature_size) = shape.get(1) else {
        return Err(Error::ModelLoad(format!(
            "encoder audio_signal input must be rank 3, got shape {shape:?}"
        )));
    };
    if feature_size <= 0 {
        return Err(Error::ModelLoad(format!(
            "encoder audio_signal feature dimension must be static, got shape {shape:?}"
        )));
    }

    Ok(feature_size as usize)
}

fn extract_state(value: &ort::value::Value, name: &str) -> Result<Array3<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::Backend(format!("failed to extract {name}: {err}")))?;
    let dims = shape.as_ref();
    if dims.len() != 3 {
        return Err(Error::Backend(format!(
            "expected 3D {name}, got shape {dims:?}"
        )));
    }
    Array3::from_shape_vec(
        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
        data.to_vec(),
    )
    .map_err(|err| Error::Backend(format!("failed to create {name}: {err}")))
}

fn find_encoder(dir: &Path) -> Result<PathBuf> {
    find_model_file(
        dir,
        &[
            "encoder.onnx",
            "encoder-model.onnx",
            "encoder-model.int8.onnx",
        ],
        "encoder",
    )
}

fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
    find_model_file(
        dir,
        &[
            "decoder_joint.onnx",
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder-model.onnx",
        ],
        "decoder_joint",
    )
}

fn find_model_file(dir: &Path, candidates: &[&str], label: &str) -> Result<PathBuf> {
    for candidate in candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(Error::ModelLoad(format!(
        "No {label} model found in {}",
        dir.display()
    )))
}

fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)
        .map_err(|err| Error::ModelLoad(format!("failed to open vocab.txt: {err}")))?;
    let reader = BufReader::new(file);
    let mut vocab = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|err| Error::ModelLoad(format!("failed to read vocab: {err}")))?;
        let Some((token, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let id = id
            .parse::<usize>()
            .map_err(|err| Error::ModelLoad(format!("invalid vocab id '{id}': {err}")))?;
        if id >= vocab.len() {
            vocab.resize(id + 1, String::new());
        }
        vocab[id] = token.to_string();
    }

    if vocab.is_empty() {
        return Err(Error::ModelLoad("vocab.txt is empty".to_string()));
    }
    Ok(vocab)
}

fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(audio.len());
    out.push(audio[0]);
    for index in 1..audio.len() {
        out.push(audio[index] - coef * audio[index - 1]);
    }
    out
}

fn stft_with_plan(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let window = hann_window(win_length);
    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
    let mut input = vec![0.0; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        input.fill(0.0);
        for sample_idx in 0..win_length.min(padded.len() - start) {
            input[sample_idx] = padded[start + sample_idx] * window[sample_idx];
        }

        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|err| Error::Backend(format!("FFT failed: {err}")))?;
        for freq_idx in 0..freq_bins {
            spectrogram[[freq_idx, frame_idx]] = output[freq_idx].norm_sqr();
        }
    }

    Ok(spectrogram)
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|index| 0.5 - 0.5 * ((2.0 * PI * index as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);
    let mel_points = (0..=n_mels + 1)
        .map(|index| {
            mel_to_hz_slaney(mel_min + (mel_max - mel_min) * index as f64 / (n_mels + 1) as f64)
        })
        .collect::<Vec<_>>();
    let fft_freqs = (0..freq_bins)
        .map(|index| index as f64 * sample_rate as f64 / n_fft as f64)
        .collect::<Vec<_>>();
    let fdiff = mel_points
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();

    for mel_idx in 0..n_mels {
        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[mel_idx]) / fdiff[mel_idx];
            let upper = (mel_points[mel_idx + 2] - freq) / fdiff[mel_idx + 1];
            filterbank[[mel_idx, freq_idx]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    for mel_idx in 0..n_mels {
        let enorm = 2.0 / (mel_points[mel_idx + 2] - mel_points[mel_idx]);
        for freq_idx in 0..freq_bins {
            filterbank[[mel_idx, freq_idx]] *= enorm as f32;
        }
    }

    filterbank
}

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_boundary_uses_spaces_for_multilingual_vocab() {
        let vocab = vec!["<unk>".to_string(), "<|en|>".to_string()];
        assert_eq!(word_boundary_for_vocab(&vocab), WordBoundary::Space);
    }

    #[test]
    fn word_boundary_strips_spaces_for_legacy_vocab() {
        let vocab = vec!["<unk>".to_string(), "token".to_string()];
        assert_eq!(word_boundary_for_vocab(&vocab), WordBoundary::Strip);
    }

    #[test]
    #[ignore = "requires the local converted Japanese TDT model"]
    fn converted_ja_model_runs_one_second_of_audio() {
        let model_dir = Path::new("models/parakeet-tdt_ctc-0.6b-ja-onnx");
        assert!(
            model_dir.exists(),
            "missing converted model at {}",
            model_dir.display()
        );

        let mut model = ParakeetTdtModel::load(model_dir, Device::Cpu).unwrap();
        let text = model.transcribe_samples(vec![0.0; PCM_SAMPLE_RATE_HZ as usize]);
        assert!(text.is_ok());
    }
}
