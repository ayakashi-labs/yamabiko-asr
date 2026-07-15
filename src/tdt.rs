use crate::{Device, Error, PCM_SAMPLE_RATE_HZ, Result};
use ndarray::{Array2, Array3, Axis};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};
use ort::value::{DynValue, Outlet, TensorElementType, TensorRef, ValueType};
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
const DECODER_STATE_LAYERS: usize = 2;
const DECODER_STATE_SIZE: usize = 640;
const SENTENCEPIECE_SPACE: char = '\u{2581}';

struct EncoderContract {
    feature_size: usize,
    encoded_size: Option<usize>,
}

pub(crate) struct ParakeetTdtModel {
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
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
        let vocab = load_vocab(&model_dir.join("vocab.txt"))?;
        let encoder_contract = validate_encoder_contract(&encoder)?;
        validate_decoder_joint_contract(&decoder_joint, &encoder_contract, vocab.len())?;
        let word_boundary = word_boundary_for_vocab(&vocab);
        let mel_basis = create_mel_filterbank(
            N_FFT,
            encoder_contract.feature_size,
            PCM_SAMPLE_RATE_HZ as usize,
        );
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);
        let window = hann_window(WIN_LENGTH);

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
            mel_basis,
            fft_plan,
            window,
            word_boundary,
        })
    }

    pub(crate) fn transcribe_samples(&mut self, mut samples: Vec<f32>) -> Result<String> {
        apply_preemphasis_in_place(&mut samples, PREEMPHASIS);
        let (features, valid_frames) = self.extract_features(&samples)?;
        let tokens = self.run_encoder_and_decode(&features, valid_frames)?;
        Ok(self.decode_tokens(&tokens))
    }

    fn extract_features(&self, audio: &[f32]) -> Result<(Array2<f32>, usize)> {
        let spectrogram = stft_with_plan(audio, &self.fft_plan, &self.window, N_FFT, HOP_LENGTH)?;
        let mut mel_spectrogram = self.mel_basis.dot(&spectrogram);
        let log_zero_guard = 2.0f32.powi(-24);
        mel_spectrogram.mapv_inplace(|value| (value + log_zero_guard).ln());

        let valid_frames = valid_feature_frames(audio.len());
        normalize_features(&mut mel_spectrogram, valid_frames)?;
        Ok((mel_spectrogram, valid_frames))
    }

    fn run_encoder_and_decode(
        &mut self,
        features: &Array2<f32>,
        valid_frames: usize,
    ) -> Result<Vec<usize>> {
        let input = features.view().insert_axis(Axis(0));
        let input_length = [i64::try_from(valid_frames)
            .map_err(|_| Error::Backend("feature length exceeds i64".to_string()))?];
        let vocab_size = self.vocab.len();
        let encoder = &mut self.encoder;
        let decoder_joint = &mut self.decoder_joint;

        let outputs = encoder
            .run(ort::inputs!(
                "audio_signal" => TensorRef::from_array_view(input)
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "length" => TensorRef::from_array_view(([1usize], &input_length[..]))
                    .map_err(|err| Error::Backend(err.to_string()))?
            ))
            .map_err(|err| Error::Backend(err.to_string()))?;

        let encoder_out = required_output(&outputs, "encoder", "outputs")?;
        let encoder_lens = required_output(&outputs, "encoder", "encoded_lengths")?;
        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|err| Error::Backend(format!("failed to extract encoder output: {err}")))?;
        let (lens_shape, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|err| Error::Backend(format!("failed to extract encoder length: {err}")))?;

        let shape_dims = shape.as_ref();
        let [batch_size, encoder_dim, encoded_frames] = shape_dims else {
            return Err(Error::Backend(format!(
                "expected 3D encoder output, got shape {shape_dims:?}"
            )));
        };
        let batch_size = usize::try_from(*batch_size)
            .map_err(|_| Error::Backend(format!("invalid encoder batch size: {batch_size}")))?;
        let encoder_dim = usize::try_from(*encoder_dim)
            .map_err(|_| Error::Backend(format!("invalid encoder dimension: {encoder_dim}")))?;
        let encoded_frames = usize::try_from(*encoded_frames).map_err(|_| {
            Error::Backend(format!("invalid encoder frame count: {encoded_frames}"))
        })?;
        if batch_size != 1 {
            return Err(Error::Backend(format!(
                "expected encoder batch size 1, got {batch_size}"
            )));
        }
        let expected_values = encoder_dim
            .checked_mul(encoded_frames)
            .ok_or_else(|| Error::Backend("encoder output size overflow".to_string()))?;
        if data.len() != expected_values {
            return Err(Error::Backend(format!(
                "encoder output contains {} values, expected {expected_values}",
                data.len()
            )));
        }
        if lens_shape.as_ref() != [1] || lens_data.len() != 1 {
            return Err(Error::Backend(format!(
                "expected encoder length shape [1], got {:?}",
                lens_shape.as_ref()
            )));
        }
        let encoder_len = usize::try_from(lens_data[0])
            .map_err(|_| Error::Backend(format!("invalid encoder length: {}", lens_data[0])))?;
        if encoder_len > encoded_frames {
            return Err(Error::Backend(format!(
                "encoder length {encoder_len} exceeds output frame count {encoded_frames}"
            )));
        }

        greedy_decode(
            decoder_joint,
            data,
            encoder_dim,
            encoded_frames,
            encoder_len,
            vocab_size,
        )
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

fn valid_feature_frames(sample_count: usize) -> usize {
    sample_count / HOP_LENGTH
}

fn normalize_features(features: &mut Array2<f32>, valid_frames: usize) -> Result<()> {
    let total_frames = features.shape()[1];
    if valid_frames > total_frames {
        return Err(Error::Backend(format!(
            "feature length {valid_frames} exceeds tensor length {total_frames}"
        )));
    }
    for mut feature in features.rows_mut() {
        if valid_frames == 0 {
            feature.fill(0.0);
            continue;
        }

        let mean = feature.iter().take(valid_frames).sum::<f32>() / valid_frames as f32;
        let variance = if valid_frames > 1 {
            feature
                .iter()
                .take(valid_frames)
                .map(|&value| (value - mean).powi(2))
                .sum::<f32>()
                / (valid_frames as f32 - 1.0)
        } else {
            0.0
        };
        let std = variance.sqrt() + 1e-5;
        for (frame, value) in feature.iter_mut().enumerate() {
            if frame < valid_frames {
                *value = (*value - mean) / std;
            } else {
                *value = 0.0;
            }
        }
    }
    Ok(())
}

fn greedy_decode(
    decoder_joint: &mut Session,
    encoder_out: &[f32],
    encoder_dim: usize,
    encoded_frames: usize,
    time_steps: usize,
    vocab_size: usize,
) -> Result<Vec<usize>> {
    let blank_id = vocab_size
        .checked_sub(1)
        .ok_or_else(|| Error::Backend("decoder vocabulary is empty".to_string()))?;
    let max_tokens_per_step = 10;

    let mut state_h = Array3::<f32>::zeros((DECODER_STATE_LAYERS, 1, DECODER_STATE_SIZE));
    let mut state_c = Array3::<f32>::zeros((DECODER_STATE_LAYERS, 1, DECODER_STATE_SIZE));
    let mut frame = vec![0.0; encoder_dim];
    let mut tokens = Vec::new();
    let mut emitted_tokens = 0;
    let mut last_emitted_token = i32::try_from(blank_id)
        .map_err(|_| Error::Backend("vocabulary is too large for decoder input".to_string()))?;
    let mut t = 0;
    let target_length = [1i32];

    while t < time_steps {
        for (dimension, value) in frame.iter_mut().enumerate() {
            *value = encoder_out[dimension * encoded_frames + t];
        }
        let targets = [last_emitted_token];

        let outputs = decoder_joint
            .run(ort::inputs!(
                "encoder_outputs" => TensorRef::from_array_view((
                    [1usize, encoder_dim, 1],
                    frame.as_slice(),
                ))
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "targets" => TensorRef::from_array_view(([1usize, 1], &targets[..]))
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "target_length" => TensorRef::from_array_view((
                    [1usize],
                    &target_length[..],
                ))
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "input_states_1" => TensorRef::from_array_view(state_h.view())
                    .map_err(|err| Error::Backend(err.to_string()))?,
                "input_states_2" => TensorRef::from_array_view(state_c.view())
                    .map_err(|err| Error::Backend(err.to_string()))?
            ))
            .map_err(|err| Error::Backend(err.to_string()))?;

        let logits = required_output(&outputs, "decoder", "outputs")?;
        let (logits_shape, logits_data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|err| Error::Backend(format!("failed to extract decoder logits: {err}")))?;
        let logits_dims = logits_shape.as_ref();
        let logits_size = decoder_logits_size(logits_dims, vocab_size, false)
            .map_err(Error::Backend)?
            .ok_or_else(|| Error::Backend("decoder returned dynamic logits shape".to_string()))?;
        if logits_data.len() != logits_size {
            return Err(Error::Backend(format!(
                "decoder returned {} logits with shape {logits_dims:?}; expected {logits_size}",
                logits_data.len()
            )));
        }
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
            let output_state_h = required_output(&outputs, "decoder", "output_states_1")?;
            let output_state_c = required_output(&outputs, "decoder", "output_states_2")?;
            copy_state(output_state_h, "output_states_1", &mut state_h)?;
            copy_state(output_state_c, "output_states_2", &mut state_c)?;
            tokens.push(token_id);
            last_emitted_token = i32::try_from(token_id)
                .map_err(|_| Error::Backend("token id exceeds decoder input range".to_string()))?;
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

fn validate_encoder_contract(encoder: &Session) -> Result<EncoderContract> {
    let audio_shape = require_tensor(
        encoder.inputs(),
        "encoder input",
        "audio_signal",
        TensorElementType::Float32,
        Some(3),
        &[(0, 1)],
    )?;
    require_tensor(
        encoder.inputs(),
        "encoder input",
        "length",
        TensorElementType::Int64,
        Some(1),
        &[(0, 1)],
    )?;
    let encoded_shape = require_tensor(
        encoder.outputs(),
        "encoder output",
        "outputs",
        TensorElementType::Float32,
        Some(3),
        &[(0, 1)],
    )?;
    require_tensor(
        encoder.outputs(),
        "encoder output",
        "encoded_lengths",
        TensorElementType::Int64,
        Some(1),
        &[(0, 1)],
    )?;

    let feature_size = known_dimension(audio_shape, 1, "encoder audio_signal feature")?
        .ok_or_else(|| {
            Error::ModelLoad(format!(
                "encoder audio_signal feature dimension must be static, got {audio_shape:?}"
            ))
        })?;
    let encoded_size = known_dimension(encoded_shape, 1, "encoder output feature")?;

    Ok(EncoderContract {
        feature_size,
        encoded_size,
    })
}

fn validate_decoder_joint_contract(
    decoder_joint: &Session,
    encoder: &EncoderContract,
    vocab_size: usize,
) -> Result<()> {
    let encoder_input_shape = require_tensor(
        decoder_joint.inputs(),
        "decoder input",
        "encoder_outputs",
        TensorElementType::Float32,
        Some(3),
        &[(0, 1), (2, 1)],
    )?;
    require_tensor(
        decoder_joint.inputs(),
        "decoder input",
        "targets",
        TensorElementType::Int32,
        Some(2),
        &[(0, 1), (1, 1)],
    )?;
    require_tensor(
        decoder_joint.inputs(),
        "decoder input",
        "target_length",
        TensorElementType::Int32,
        Some(1),
        &[(0, 1)],
    )?;
    for name in ["input_states_1", "input_states_2"] {
        require_tensor(
            decoder_joint.inputs(),
            "decoder input",
            name,
            TensorElementType::Float32,
            Some(3),
            &[(0, DECODER_STATE_LAYERS), (1, 1), (2, DECODER_STATE_SIZE)],
        )?;
    }

    let logits_shape = require_tensor(
        decoder_joint.outputs(),
        "decoder output",
        "outputs",
        TensorElementType::Float32,
        None,
        &[],
    )?;
    decoder_logits_size(logits_shape, vocab_size, true).map_err(Error::ModelLoad)?;
    for name in ["output_states_1", "output_states_2"] {
        require_tensor(
            decoder_joint.outputs(),
            "decoder output",
            name,
            TensorElementType::Float32,
            Some(3),
            &[(0, DECODER_STATE_LAYERS), (1, 1), (2, DECODER_STATE_SIZE)],
        )?;
    }

    if let Some(encoded_size) = encoder.encoded_size {
        require_compatible_dimension(
            encoder_input_shape,
            1,
            encoded_size,
            "decoder encoder_outputs feature",
        )?;
    }
    Ok(())
}

fn decoder_logits_size(
    shape: &[i64],
    vocab_size: usize,
    allow_dynamic: bool,
) -> std::result::Result<Option<usize>, String> {
    let Some((&last_dimension, leading_dimensions)) = shape.split_last() else {
        return Err("decoder logits must not be a scalar".to_string());
    };
    if leading_dimensions
        .iter()
        .any(|&dimension| dimension != 1 && !(allow_dynamic && dimension == -1))
    {
        return Err(format!(
            "decoder logits must contain one prediction, got shape {shape:?}"
        ));
    }

    let logits_size = match last_dimension {
        -1 if allow_dynamic => None,
        value if value > 0 => Some(
            usize::try_from(value)
                .map_err(|_| format!("decoder logits dimension is too large: {value}"))?,
        ),
        value => {
            return Err(format!(
                "decoder logits dimension must be positive{}, got {value}",
                if allow_dynamic { " or dynamic" } else { "" }
            ));
        }
    };
    let minimum_logits = vocab_size
        .checked_add(1)
        .ok_or_else(|| "decoder logits size overflow".to_string())?;
    if let Some(logits_size) = logits_size
        && logits_size < minimum_logits
    {
        return Err(format!(
            "decoder logits dimension {logits_size} is too small for {vocab_size} vocabulary entries and a duration output"
        ));
    }
    Ok(logits_size)
}

fn require_tensor<'a>(
    outlets: &'a [Outlet],
    location: &str,
    name: &str,
    expected_type: TensorElementType,
    expected_rank: Option<usize>,
    expected_dimensions: &[(usize, usize)],
) -> Result<&'a [i64]> {
    let outlet = outlets
        .iter()
        .find(|outlet| outlet.name() == name)
        .ok_or_else(|| Error::ModelLoad(format!("{location} is missing '{name}'")))?;
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(Error::ModelLoad(format!(
            "{location} '{name}' must be a tensor, got {}",
            outlet.dtype()
        )));
    };
    if *ty != expected_type {
        return Err(Error::ModelLoad(format!(
            "{location} '{name}' must contain {expected_type}, got {ty}"
        )));
    }
    if let Some(expected_rank) = expected_rank
        && shape.len() != expected_rank
    {
        return Err(Error::ModelLoad(format!(
            "{location} '{name}' must have rank {expected_rank}, got shape {shape:?}"
        )));
    }
    for &(index, expected) in expected_dimensions {
        require_compatible_dimension(shape, index, expected, &format!("{location} '{name}'"))?;
    }
    Ok(shape.as_ref())
}

fn known_dimension(shape: &[i64], index: usize, label: &str) -> Result<Option<usize>> {
    match shape.get(index).copied() {
        Some(-1) => Ok(None),
        Some(value) if value > 0 => usize::try_from(value)
            .map(Some)
            .map_err(|_| Error::ModelLoad(format!("{label} dimension is too large: {value}"))),
        Some(value) => Err(Error::ModelLoad(format!(
            "{label} dimension must be positive or dynamic, got {value}"
        ))),
        None => Err(Error::ModelLoad(format!(
            "{label} dimension is missing from shape {shape:?}"
        ))),
    }
}

fn require_compatible_dimension(
    shape: &[i64],
    index: usize,
    expected: usize,
    label: &str,
) -> Result<()> {
    if let Some(actual) = known_dimension(shape, index, label)?
        && actual != expected
    {
        return Err(Error::ModelLoad(format!(
            "{label} dimension {actual} does not match required size {expected}"
        )));
    }
    Ok(())
}

fn required_output<'a>(
    outputs: &'a SessionOutputs<'_>,
    model: &str,
    name: &str,
) -> Result<&'a DynValue> {
    outputs
        .get(name)
        .ok_or_else(|| Error::Backend(format!("{model} returned no '{name}' value")))
}

fn copy_state(value: &ort::value::Value, name: &str, destination: &mut Array3<f32>) -> Result<()> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::Backend(format!("failed to extract {name}: {err}")))?;
    let dims = shape.as_ref();
    let expected = destination.shape();
    let expected_dims = [expected[0] as i64, expected[1] as i64, expected[2] as i64];
    if dims != expected_dims {
        return Err(Error::Backend(format!(
            "expected {name} shape {expected:?}, got {dims:?}"
        )));
    }
    let destination = destination
        .as_slice_mut()
        .ok_or_else(|| Error::Backend("decoder state array is not contiguous".to_string()))?;
    if data.len() != destination.len() {
        return Err(Error::Backend(format!(
            "{name} contains {} values, expected {}",
            data.len(),
            destination.len()
        )));
    }
    destination.copy_from_slice(data);
    Ok(())
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
    parse_vocab(BufReader::new(file))
}

fn parse_vocab(reader: impl BufRead) -> Result<Vec<String>> {
    let mut entries = Vec::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|err| Error::ModelLoad(format!("failed to read vocab: {err}")))?;
        if line.is_empty() {
            continue;
        }
        let (token, id) = line.rsplit_once(' ').ok_or_else(|| {
            Error::ModelLoad(format!(
                "invalid vocab entry on line {line_number}: expected '<token> <id>'"
            ))
        })?;
        if token.is_empty() {
            return Err(Error::ModelLoad(format!(
                "invalid empty token on vocab line {line_number}"
            )));
        }
        let id = id.parse::<usize>().map_err(|err| {
            Error::ModelLoad(format!(
                "invalid vocab id '{id}' on line {line_number}: {err}"
            ))
        })?;
        entries.push((id, line_number, token.to_string()));
    }

    if entries.is_empty() {
        return Err(Error::ModelLoad("vocab.txt is empty".to_string()));
    }
    entries.sort_unstable_by_key(|entry| entry.0);
    for (expected_id, (actual_id, line_number, _)) in entries.iter().enumerate() {
        if *actual_id < expected_id {
            return Err(Error::ModelLoad(format!(
                "duplicate vocab id {actual_id} on line {line_number}"
            )));
        }
        if *actual_id > expected_id {
            return Err(Error::ModelLoad(format!(
                "vocab.txt is missing token id {expected_id}"
            )));
        }
    }
    let vocab = entries
        .into_iter()
        .map(|(_, _, token)| token)
        .collect::<Vec<_>>();
    if let Some(blank_id) = vocab
        .iter()
        .position(|token| matches!(token.as_str(), "<blank>" | "<blk>"))
        && blank_id + 1 != vocab.len()
    {
        return Err(Error::ModelLoad(
            "the <blank> or <blk> vocab entry must be last".to_string(),
        ));
    }
    Ok(vocab)
}

fn apply_preemphasis_in_place(audio: &mut [f32], coef: f32) {
    for index in (1..audio.len()).rev() {
        audio[index] -= coef * audio[index - 1];
    }
}

fn stft_with_plan(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Result<Array2<f32>> {
    if window.len() > n_fft {
        return Err(Error::Backend(format!(
            "STFT window length {} exceeds FFT length {n_fft}",
            window.len()
        )));
    }
    let pad_amount = n_fft / 2;
    let padded_len = audio.len().saturating_add(pad_amount * 2);
    let num_frames = (padded_len - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let window_offset = (n_fft - window.len()) / 2;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
    let mut input = vec![0.0; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        input.fill(0.0);
        for (window_idx, &weight) in window.iter().enumerate() {
            let fft_idx = window_offset + window_idx;
            let padded_index = start + fft_idx;
            if let Some(audio_index) = padded_index.checked_sub(pad_amount)
                && let Some(&sample) = audio.get(audio_index)
            {
                input[fft_idx] = sample * weight;
            }
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
    use ort::value::{Shape, SymbolicDimensions};
    use std::io::Cursor;

    fn parse_test_vocab(contents: &str) -> Result<Vec<String>> {
        parse_vocab(Cursor::new(contents))
    }

    fn tensor_outlet(name: &str, ty: TensorElementType, shape: &[i64]) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty,
                shape: Shape::new(shape.iter().copied()),
                dimension_symbols: SymbolicDimensions::empty(shape.len()),
            },
        )
    }

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
    fn preemphasis_updates_samples_in_place_from_original_predecessors() {
        let mut audio = [1.0, 2.0, 3.0];
        apply_preemphasis_in_place(&mut audio, PREEMPHASIS);

        assert_eq!(audio[0], 1.0);
        assert!((audio[1] - 1.03).abs() < 1e-6);
        assert!((audio[2] - 1.06).abs() < 1e-6);
    }

    #[test]
    fn stft_centers_a_shorter_window_inside_the_fft_frame() {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(8);
        let spectrogram =
            stft_with_plan(&[1.0, 0.0, 0.0, 0.0], &plan, &[1.0, 2.0, 3.0, 4.0], 8, 4).unwrap();

        for magnitude in spectrogram.column(0) {
            assert!((*magnitude - 9.0).abs() < 1e-5);
        }
    }

    #[test]
    fn normalization_uses_only_nemo_valid_frames_and_masks_the_tail() {
        let mut features = Array2::from_shape_vec((1, 3), vec![1.0, 3.0, 100.0]).unwrap();
        normalize_features(&mut features, 2).unwrap();

        let expected = 1.0 / (2.0f32.sqrt() + 1e-5);
        assert!((features[[0, 0]] + expected).abs() < 1e-5);
        assert!((features[[0, 1]] - expected).abs() < 1e-5);
        assert_eq!(features[[0, 2]], 0.0);
        assert_eq!(valid_feature_frames(PCM_SAMPLE_RATE_HZ as usize), 100);
        assert!(normalize_features(&mut features, 4).is_err());
    }

    #[test]
    fn vocab_requires_contiguous_unique_ids_and_final_blank() {
        assert_eq!(
            parse_test_vocab("\ntoken 0\n<blank> 1\n").unwrap(),
            ["token", "<blank>"]
        );

        for (contents, expected_message) in [
            ("token\n<blank> 1\n", "invalid vocab entry"),
            ("token 0\nother 0\n<blank> 1\n", "duplicate vocab id 0"),
            ("token 0\n<blank> 2\n", "missing token id 1"),
            ("<blank> 0\ntoken 1\n", "must be last"),
        ] {
            let error = parse_test_vocab(contents).unwrap_err();
            assert!(
                error.to_string().contains(expected_message),
                "unexpected error: {error}"
            );
        }

        let huge_id = format!("token {}\n", usize::MAX);
        assert!(
            parse_test_vocab(&huge_id)
                .unwrap_err()
                .to_string()
                .contains("missing token id 0")
        );
    }

    #[test]
    fn model_dimensions_accept_dynamic_and_reject_invalid_values() {
        assert_eq!(known_dimension(&[-1, 80, -1], 0, "test").unwrap(), None);
        assert_eq!(known_dimension(&[-1, 80, -1], 1, "test").unwrap(), Some(80));
        assert!(known_dimension(&[-1, 0, -1], 1, "test").is_err());
        assert!(known_dimension(&[-1, -2, -1], 1, "test").is_err());
    }

    #[test]
    fn decoder_logits_contract_uses_the_final_dimension_at_any_rank() {
        for shape in [&[7][..], &[1, 1, 7], &[1, 1, 1, 1, 7]] {
            assert_eq!(decoder_logits_size(shape, 5, true).unwrap(), Some(7));
            assert_eq!(decoder_logits_size(shape, 5, false).unwrap(), Some(7));
        }

        assert_eq!(decoder_logits_size(&[1, -1], 5, true).unwrap(), None);
        assert!(decoder_logits_size(&[1, -1], 5, false).is_err());
        assert!(decoder_logits_size(&[], 5, true).is_err());
        assert!(decoder_logits_size(&[2, 7], 5, true).is_err());
        assert!(decoder_logits_size(&[1, 5], 5, true).is_err());
    }

    #[test]
    fn tensor_contract_checks_name_type_rank_and_static_dimensions() {
        let valid = [tensor_outlet(
            "audio_signal",
            TensorElementType::Float32,
            &[-1, 80, -1],
        )];
        assert_eq!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(3),
                &[(0, 1)],
            )
            .unwrap(),
            [-1, 80, -1]
        );

        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "missing",
                TensorElementType::Float32,
                Some(3),
                &[],
            )
            .is_err()
        );
        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Int32,
                Some(3),
                &[],
            )
            .is_err()
        );
        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(2),
                &[],
            )
            .is_err()
        );

        let incompatible = [tensor_outlet(
            "audio_signal",
            TensorElementType::Float32,
            &[2, 80, -1],
        )];
        assert!(
            require_tensor(
                &incompatible,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(3),
                &[(0, 1)],
            )
            .is_err()
        );
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
