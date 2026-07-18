// Parts of the streaming-state update and cache-compression logic in this
// module are derived from NVIDIA NeMo.
// Copyright (c) 2020-2026 NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0.

use crate::audio_features::{StftWorkspace, hann_window, slaney_mel_filterbank};
use crate::diarization::{
    BackendSpeakerId, DiarizationOutput, DiarizedRegion, Diarizer, DiarizerFactory,
};
use crate::ort_utils;
use crate::{AudioSourceId, Device, Error, PCM_SAMPLE_RATE_HZ, Result};
use ndarray::{Array2, Array3};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, TensorElementType, TensorRef};
use realfft::RealToComplex;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

const MAX_SPEAKERS: usize = 4;
const FP32_MODEL_FILENAME: &str = "sortformer.fp32.onnx";
const FP16_MODEL_FILENAME: &str = "sortformer.fp16.onnx";
const MODEL_KIND: &str = "sortformer_streaming_diarization";
const MODEL_ID: &str = "nvidia/diar_streaming_sortformer_4spk-v2.1";
const MODEL_REVISION: &str = "a494724e2261b51d18a6ef403343b1f7025b3b6d";
const NEMO_VERSION: &str = "2.7.3";
const CONTRACT_VERSION: &str = "1";

const FEATURE_SIZE: usize = 128;
const EMBEDDING_SIZE: usize = 512;
const SUBSAMPLING_FACTOR: usize = 8;
const DIARIZATION_FRAME_SAMPLES: u64 = 1_280;

const CHUNK_LEN: usize = 6;
const RIGHT_CONTEXT: usize = 7;
const LEFT_CONTEXT: usize = 1;
const FIFO_LEN: usize = 188;
const UPDATE_PERIOD: usize = 144;
const SPEAKER_CACHE_LEN: usize = 188;
const SILENCE_FRAMES_PER_SPEAKER: usize = 3;

const MAIN_FEATURE_FRAMES: usize = CHUNK_LEN * SUBSAMPLING_FACTOR;
const RIGHT_FEATURE_FRAMES: usize = RIGHT_CONTEXT * SUBSAMPLING_FACTOR;
const LEFT_FEATURE_FRAMES: usize = LEFT_CONTEXT * SUBSAMPLING_FACTOR;
const DEFAULT_WIN_LENGTH: usize = 400;
const MODEL_LOOKAHEAD_SAMPLES: usize =
    (CHUNK_LEN + RIGHT_CONTEXT) * DIARIZATION_FRAME_SAMPLES as usize + DEFAULT_WIN_LENGTH / 2;

const ONSET: f32 = 0.5;
const OFFSET: f32 = 0.5;
const SILENCE_THRESHOLD: f32 = 0.2;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const SCORES_BOOST_LATEST: f32 = 0.05;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;

type Embedding = [f32; EMBEDDING_SIZE];
type Prediction = [f32; MAX_SPEAKERS];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelPrecision {
    Fp32,
    Fp16,
}

impl ModelPrecision {
    const fn filename(self) -> &'static str {
        match self {
            Self::Fp32 => FP32_MODEL_FILENAME,
            Self::Fp16 => FP16_MODEL_FILENAME,
        }
    }

    const fn metadata_value(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Cpu,
    Cuda,
    DirectMl,
}

impl ProviderKind {
    const fn device(self) -> Device {
        match self {
            Self::Cpu => Device::Cpu,
            Self::Cuda => Device::Cuda,
            Self::DirectMl => Device::DirectMl,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::DirectMl => "DirectML",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelCandidate {
    provider: ProviderKind,
    precision: ModelPrecision,
}

impl ModelCandidate {
    const fn strict(self) -> bool {
        !matches!(self.provider, ProviderKind::Cpu)
    }
}

pub(crate) struct SortformerDiarizerFactory {
    model_dir: PathBuf,
    device: Device,
    max_sources: usize,
}

impl SortformerDiarizerFactory {
    pub(crate) fn new(model_dir: PathBuf, device: Device, max_sources: usize) -> Result<Self> {
        retained_pcm_limit(max_sources)?;
        Ok(Self {
            model_dir,
            device,
            max_sources,
        })
    }
}

impl DiarizerFactory for SortformerDiarizerFactory {
    fn create(&mut self) -> Result<Box<dyn Diarizer>> {
        SortformerDiarizer::load(&self.model_dir, self.device, self.max_sources)
            .map(|diarizer| Box::new(diarizer) as Box<dyn Diarizer>)
    }
}

struct SortformerDiarizer {
    session: Session,
    frontend: Arc<Frontend>,
    sources: HashMap<AudioSourceId, SortformerSource>,
    max_sources: usize,
    max_retained_samples: usize,
}

impl SortformerDiarizer {
    fn load(model_dir: &Path, device: Device, max_sources: usize) -> Result<Self> {
        let candidates = model_candidates(device)?;
        if !model_dir.is_dir() {
            return Err(Error::DiarizationModelLoad(format!(
                "Sortformer model path must be a directory: {}",
                model_dir.display()
            )));
        }

        let mut failures = Vec::new();
        for candidate in candidates {
            match load_candidate(model_dir, candidate) {
                Ok((session, frontend)) => {
                    return Ok(Self {
                        session,
                        frontend: Arc::new(frontend),
                        sources: HashMap::new(),
                        max_sources,
                        max_retained_samples: retained_pcm_limit(max_sources)
                            .expect("factory validated retained PCM limit"),
                    });
                }
                Err(err) if device == Device::Auto => failures.push(format!(
                    "{} {}: {err}",
                    candidate.provider.label(),
                    candidate.precision.metadata_value()
                )),
                Err(err) => return Err(err),
            }
        }

        Err(Error::DiarizationModelLoad(format!(
            "no usable Sortformer model/provider candidate: {}",
            failures.join("; ")
        )))
    }
}

fn retained_pcm_limit(max_sources: usize) -> Result<usize> {
    if max_sources == 0 {
        return Err(Error::InvalidConfig(
            "diarization max_sources must be greater than zero".to_string(),
        ));
    }
    MODEL_LOOKAHEAD_SAMPLES
        .checked_mul(max_sources)
        .ok_or_else(|| {
            Error::InvalidConfig("diarization retained PCM limit overflowed".to_string())
        })
}

impl Diarizer for SortformerDiarizer {
    fn max_retained_samples(&self) -> usize {
        self.max_retained_samples
    }

    fn open_source(&mut self, source_id: AudioSourceId) -> Result<()> {
        if self.sources.contains_key(&source_id) {
            return Err(Error::InvalidConfig(format!(
                "diarization source {} is already active",
                source_id.get()
            )));
        }
        if self.sources.len() >= self.max_sources {
            return Err(Error::SourceLimit {
                max_sources: self.max_sources,
            });
        }
        self.sources
            .insert(source_id, SortformerSource::new(Arc::clone(&self.frontend)));
        Ok(())
    }

    fn push(
        &mut self,
        source_id: AudioSourceId,
        samples: &[f32],
        start_sample: u64,
        cancelled: &AtomicBool,
    ) -> Result<DiarizationOutput> {
        let source = self
            .sources
            .get_mut(&source_id)
            .ok_or(Error::SourceNotFound { source_id })?;
        source.push(&mut self.session, samples, start_sample, cancelled)
    }

    fn finish(
        &mut self,
        source_id: AudioSourceId,
        cancelled: &AtomicBool,
    ) -> Result<DiarizationOutput> {
        let output = self
            .sources
            .get_mut(&source_id)
            .ok_or(Error::SourceNotFound { source_id })?
            .finish(&mut self.session, cancelled)?;
        self.sources.remove(&source_id);
        Ok(output)
    }
}

fn model_candidates(device: Device) -> Result<Vec<ModelCandidate>> {
    let cpu = ModelCandidate {
        provider: ProviderKind::Cpu,
        precision: ModelPrecision::Fp32,
    };
    let cuda = ModelCandidate {
        provider: ProviderKind::Cuda,
        precision: ModelPrecision::Fp16,
    };
    let directml = ModelCandidate {
        provider: ProviderKind::DirectMl,
        precision: ModelPrecision::Fp16,
    };

    match device {
        Device::Cpu => Ok(vec![cpu]),
        Device::Auto => Ok(vec![
            #[cfg(feature = "cuda")]
            cuda,
            #[cfg(feature = "directml")]
            directml,
            cpu,
        ]),
        Device::Cuda if cfg!(feature = "cuda") => Ok(vec![cuda]),
        Device::DirectMl if cfg!(feature = "directml") => Ok(vec![directml]),
        Device::Cuda => Err(provider_feature_disabled(device, "cuda")),
        Device::DirectMl => Err(provider_feature_disabled(device, "directml")),
        _ => Err(Error::DeviceUnavailable {
            device,
            message: "Streaming Sortformer supports only Auto, CPU, CUDA, and DirectML".to_string(),
        }),
    }
}

fn provider_feature_disabled(device: Device, feature: &str) -> Error {
    Error::DeviceUnavailable {
        device,
        message: format!("the '{feature}' Cargo feature is not enabled"),
    }
}

fn load_candidate(model_dir: &Path, candidate: ModelCandidate) -> Result<(Session, Frontend)> {
    let model_path = model_dir.join(candidate.precision.filename());
    if !model_path.is_file() {
        return Err(Error::DiarizationModelLoad(format!(
            "No {} found in {}",
            candidate.precision.filename(),
            model_dir.display()
        )));
    }

    let provider = execution_provider(candidate.provider);
    let session = ort_utils::build_session_with_options(
        &model_path,
        candidate.provider.device(),
        vec![provider.error_on_failure()],
        candidate.strict(),
        Error::DiarizationModelLoad,
    )?;
    validate_model_contract(&session, candidate.precision)?;
    let frontend = Frontend::from_metadata(&session)?;
    Ok((session, frontend))
}

fn execution_provider(provider: ProviderKind) -> ExecutionProviderDispatch {
    match provider {
        ProviderKind::Cpu => ort::ep::CPU::default().build(),
        ProviderKind::Cuda => ort::ep::CUDA::default().build(),
        ProviderKind::DirectMl => ort::ep::DirectML::default().build(),
    }
}

struct SortformerSource {
    features: StreamingFeatures,
    cache: StreamingCache,
    next_main_feature: u64,
    next_diarization_frame: u64,
    finalized_until_sample: u64,
    active_speakers: [bool; MAX_SPEAKERS],
    finished: bool,
}

impl SortformerSource {
    fn new(frontend: Arc<Frontend>) -> Self {
        Self {
            features: StreamingFeatures::new(frontend),
            cache: StreamingCache::default(),
            next_main_feature: 0,
            next_diarization_frame: 0,
            finalized_until_sample: 0,
            active_speakers: [false; MAX_SPEAKERS],
            finished: false,
        }
    }

    fn push(
        &mut self,
        session: &mut Session,
        samples: &[f32],
        start_sample: u64,
        cancelled: &AtomicBool,
    ) -> Result<DiarizationOutput> {
        if self.finished {
            return Err(Error::Diarization(
                "cannot push audio after diarization has finished".to_string(),
            ));
        }
        if start_sample != self.features.received_samples() {
            return Err(Error::Diarization(format!(
                "non-contiguous diarization audio: expected sample {}, got {start_sample}",
                self.features.received_samples()
            )));
        }
        if samples.is_empty() {
            return Ok(self.empty_output());
        }

        self.features.push(samples)?;
        let regions = self.process_ready_windows(session, false, cancelled)?;
        let finalized = self
            .next_diarization_frame
            .saturating_mul(DIARIZATION_FRAME_SAMPLES)
            .min(self.features.received_samples());
        self.finalized_until_sample = self.finalized_until_sample.max(finalized);
        Ok(DiarizationOutput {
            regions,
            finalized_until: self.finalized_until_sample,
        })
    }

    fn finish(
        &mut self,
        session: &mut Session,
        cancelled: &AtomicBool,
    ) -> Result<DiarizationOutput> {
        if self.finished {
            return Err(Error::Diarization(
                "diarization source was already finished".to_string(),
            ));
        }
        self.finished = true;
        self.features.finish()?;
        let total_samples = self.features.received_samples();
        let mut regions = self.process_ready_windows(session, true, cancelled)?;
        let predicted_until = self
            .next_diarization_frame
            .saturating_mul(DIARIZATION_FRAME_SAMPLES)
            .min(total_samples);
        self.finalized_until_sample = self.finalized_until_sample.max(predicted_until);
        if self.finalized_until_sample < total_samples {
            append_region(
                &mut regions,
                self.finalized_until_sample,
                total_samples,
                [false; MAX_SPEAKERS],
            );
        }
        self.finalized_until_sample = total_samples;
        Ok(DiarizationOutput {
            regions,
            finalized_until: total_samples,
        })
    }

    fn process_ready_windows(
        &mut self,
        session: &mut Session,
        finishing: bool,
        cancelled: &AtomicBool,
    ) -> Result<Vec<DiarizedRegion>> {
        let mut regions = Vec::new();
        loop {
            if cancelled.load(AtomicOrdering::Acquire) {
                break;
            }
            let available_end = self.features.available_end();
            if finishing {
                if self.next_main_feature >= available_end {
                    break;
                }
            } else {
                let required_end = self
                    .next_main_feature
                    .checked_add((MAIN_FEATURE_FRAMES + RIGHT_FEATURE_FRAMES) as u64)
                    .ok_or_else(|| Error::Diarization("feature timeline overflow".to_string()))?;
                if available_end < required_end {
                    break;
                }
            }

            let left = self.next_main_feature.min(LEFT_FEATURE_FRAMES as u64) as usize;
            let real_main =
                (available_end - self.next_main_feature).min(MAIN_FEATURE_FRAMES as u64) as usize;
            let real_right = (available_end
                - self.next_main_feature.saturating_add(real_main as u64))
            .min(RIGHT_FEATURE_FRAMES as u64) as usize;
            let window_start = self.next_main_feature.saturating_sub(left as u64);
            let real_end = self
                .next_main_feature
                .saturating_add(real_main as u64)
                .saturating_add(real_right as u64);
            let mut window = self.features.copy_range(window_start, real_end)?;
            let desired_frames = padded_window_frames(left, real_main);
            window.resize(desired_frames * FEATURE_SIZE, 0.0);

            let inference = run_window(session, &self.cache, &window, desired_frames)?;
            if cancelled.load(AtomicOrdering::Acquire) {
                break;
            }
            let predictions =
                self.cache
                    .update(inference, left / SUBSAMPLING_FACTOR, RIGHT_CONTEXT)?;
            let first_prediction = self.next_diarization_frame;
            let real_sample_limit = self.features.received_samples();
            for prediction in predictions {
                let frame_start = self
                    .next_diarization_frame
                    .checked_mul(DIARIZATION_FRAME_SAMPLES)
                    .ok_or_else(|| {
                        Error::Diarization("diarization timeline overflow".to_string())
                    })?;
                if frame_start >= real_sample_limit {
                    break;
                }
                let frame_end = frame_start
                    .saturating_add(DIARIZATION_FRAME_SAMPLES)
                    .min(real_sample_limit);
                apply_activity_threshold(&mut self.active_speakers, prediction);
                append_region(&mut regions, frame_start, frame_end, self.active_speakers);
                self.next_diarization_frame = self.next_diarization_frame.saturating_add(1);
            }

            self.next_main_feature = self
                .next_main_feature
                .saturating_add(MAIN_FEATURE_FRAMES as u64);
            self.features.discard_before(
                self.next_main_feature
                    .saturating_sub(LEFT_FEATURE_FRAMES as u64),
            );

            if self.next_diarization_frame == first_prediction && finishing {
                return Err(Error::Diarization(
                    "Sortformer produced no prediction for a non-empty final window".to_string(),
                ));
            }
        }
        Ok(regions)
    }

    fn empty_output(&self) -> DiarizationOutput {
        DiarizationOutput {
            regions: Vec::new(),
            finalized_until: self.finalized_until_sample,
        }
    }
}

fn padded_window_frames(left: usize, real_main: usize) -> usize {
    left + real_main + RIGHT_FEATURE_FRAMES
}

fn apply_activity_threshold(active_speakers: &mut [bool; MAX_SPEAKERS], prediction: Prediction) {
    for (speaker, probability) in prediction.into_iter().enumerate() {
        active_speakers[speaker] = if active_speakers[speaker] {
            probability >= OFFSET
        } else {
            probability > ONSET
        };
    }
}

fn append_region(
    regions: &mut Vec<DiarizedRegion>,
    start_sample: u64,
    end_sample: u64,
    active_speakers: [bool; MAX_SPEAKERS],
) {
    if start_sample >= end_sample {
        return;
    }
    let speakers = active_speakers
        .into_iter()
        .enumerate()
        .filter(|(_, active)| *active)
        .map(|(speaker, _)| BackendSpeakerId::new(speaker as u64))
        .collect::<Vec<_>>();
    if let Some(previous) = regions.last_mut()
        && previous.end_sample == start_sample
        && previous.speakers == speakers
    {
        previous.end_sample = end_sample;
    } else {
        regions.push(DiarizedRegion {
            start_sample,
            end_sample,
            speakers,
        });
    }
}

fn run_window(
    session: &mut Session,
    cache: &StreamingCache,
    features: &[f32],
    feature_frames: usize,
) -> Result<ModelInference> {
    let chunk = Array3::from_shape_vec((1, feature_frames, FEATURE_SIZE), features.to_vec())
        .map_err(|err| Error::Diarization(format!("failed to shape feature tensor: {err}")))?;
    let spkcache = embeddings_to_array(&cache.spkcache)?;
    let fifo = embeddings_to_array(&cache.fifo)?;
    let chunk_lengths = [i64::try_from(feature_frames)
        .map_err(|_| Error::Diarization("feature length exceeds i64".to_string()))?];
    let spkcache_lengths = [i64::try_from(cache.spkcache.len())
        .map_err(|_| Error::Diarization("speaker cache length exceeds i64".to_string()))?];
    let fifo_lengths = [i64::try_from(cache.fifo.len())
        .map_err(|_| Error::Diarization("FIFO length exceeds i64".to_string()))?];

    let outputs = session
        .run(ort::inputs!(
            "chunk" => TensorRef::from_array_view(chunk.view())
                .map_err(|err| Error::Diarization(err.to_string()))?,
            "chunk_lengths" => TensorRef::from_array_view(([1usize], &chunk_lengths[..]))
                .map_err(|err| Error::Diarization(err.to_string()))?,
            "spkcache" => TensorRef::from_array_view(spkcache.view())
                .map_err(|err| Error::Diarization(err.to_string()))?,
            "spkcache_lengths" => TensorRef::from_array_view(([1usize], &spkcache_lengths[..]))
                .map_err(|err| Error::Diarization(err.to_string()))?,
            "fifo" => TensorRef::from_array_view(fifo.view())
                .map_err(|err| Error::Diarization(err.to_string()))?,
            "fifo_lengths" => TensorRef::from_array_view(([1usize], &fifo_lengths[..]))
                .map_err(|err| Error::Diarization(err.to_string()))?
        ))
        .map_err(|err| Error::Diarization(err.to_string()))?;
    extract_inference(&outputs)
}

struct ModelInference {
    predictions: Vec<Prediction>,
    chunk_embeddings: Vec<Embedding>,
    chunk_embedding_length: usize,
}

fn extract_inference(outputs: &SessionOutputs<'_>) -> Result<ModelInference> {
    let predictions = extract_predictions(required_output(outputs, "spkcache_fifo_chunk_preds")?)?;
    let chunk_embeddings = extract_embeddings(required_output(outputs, "chunk_pre_encode_embs")?)?;
    let length_value = required_output(outputs, "chunk_pre_encode_lengths")?;
    let (length_shape, length_data) = length_value
        .try_extract_tensor::<i64>()
        .map_err(|err| Error::Diarization(format!("failed to extract chunk length: {err}")))?;
    if length_shape.as_ref() != [1] || length_data.len() != 1 {
        return Err(Error::Diarization(format!(
            "expected chunk_pre_encode_lengths shape [1], got {:?}",
            length_shape.as_ref()
        )));
    }
    let chunk_embedding_length = usize::try_from(length_data[0]).map_err(|_| {
        Error::Diarization(format!(
            "invalid chunk embedding length: {}",
            length_data[0]
        ))
    })?;
    if chunk_embedding_length > chunk_embeddings.len() {
        return Err(Error::Diarization(format!(
            "chunk embedding length {chunk_embedding_length} exceeds tensor length {}",
            chunk_embeddings.len()
        )));
    }
    Ok(ModelInference {
        predictions,
        chunk_embeddings,
        chunk_embedding_length,
    })
}

fn required_output<'a>(outputs: &'a SessionOutputs<'_>, name: &str) -> Result<&'a DynValue> {
    outputs
        .get(name)
        .ok_or_else(|| Error::Diarization(format!("Sortformer returned no '{name}' output value")))
}

fn extract_predictions(value: &DynValue) -> Result<Vec<Prediction>> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|err| {
        Error::Diarization(format!("failed to extract Sortformer predictions: {err}"))
    })?;
    let [batch, frames, speakers] = shape.as_ref() else {
        return Err(Error::Diarization(format!(
            "expected prediction rank 3, got shape {:?}",
            shape.as_ref()
        )));
    };
    if *batch != 1 || *speakers != MAX_SPEAKERS as i64 {
        return Err(Error::Diarization(format!(
            "expected prediction shape [1, frames, {MAX_SPEAKERS}], got {:?}",
            shape.as_ref()
        )));
    }
    let frames = usize::try_from(*frames)
        .map_err(|_| Error::Diarization("invalid prediction frame count".to_string()))?;
    if data.len() != frames.saturating_mul(MAX_SPEAKERS) {
        return Err(Error::Diarization(format!(
            "prediction output contains {} values for {frames} frames",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(MAX_SPEAKERS)
        .map(|frame| frame.try_into().expect("prediction chunk has fixed length"))
        .collect())
}

fn extract_embeddings(value: &DynValue) -> Result<Vec<Embedding>> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|err| {
        Error::Diarization(format!("failed to extract Sortformer embeddings: {err}"))
    })?;
    let [batch, frames, dimensions] = shape.as_ref() else {
        return Err(Error::Diarization(format!(
            "expected embedding rank 3, got shape {:?}",
            shape.as_ref()
        )));
    };
    if *batch != 1 || *dimensions != EMBEDDING_SIZE as i64 {
        return Err(Error::Diarization(format!(
            "expected embedding shape [1, frames, {EMBEDDING_SIZE}], got {:?}",
            shape.as_ref()
        )));
    }
    let frames = usize::try_from(*frames)
        .map_err(|_| Error::Diarization("invalid embedding frame count".to_string()))?;
    if data.len() != frames.saturating_mul(EMBEDDING_SIZE) {
        return Err(Error::Diarization(format!(
            "embedding output contains {} values for {frames} frames",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(EMBEDDING_SIZE)
        .map(|frame| frame.try_into().expect("embedding chunk has fixed length"))
        .collect())
}

fn embeddings_to_array(embeddings: &[Embedding]) -> Result<Array3<f32>> {
    // DirectML rejects zero-sized tensor dimensions. The companion length input
    // remains zero, but a single zero frame is bound until real state exists.
    let physical_frames = embeddings.len().max(1);
    let mut values = Vec::with_capacity(physical_frames.saturating_mul(EMBEDDING_SIZE));
    for embedding in embeddings {
        values.extend_from_slice(embedding);
    }
    values.resize(physical_frames * EMBEDDING_SIZE, 0.0);
    Array3::from_shape_vec((1, physical_frames, EMBEDDING_SIZE), values)
        .map_err(|err| Error::Diarization(format!("failed to shape streaming state: {err}")))
}

struct StreamingCache {
    spkcache: Vec<Embedding>,
    spkcache_preds: Option<Vec<Prediction>>,
    fifo: Vec<Embedding>,
    fifo_preds: Vec<Prediction>,
    mean_silence_embedding: Embedding,
    silence_frames: u64,
}

impl Default for StreamingCache {
    fn default() -> Self {
        Self {
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            mean_silence_embedding: [0.0; EMBEDDING_SIZE],
            silence_frames: 0,
        }
    }
}

impl StreamingCache {
    // Mirrors NeMo 2.7.3's synchronous state update. The backend keeps one
    // batch-one state per source, so asynchronous batch padding is unnecessary.
    fn update(
        &mut self,
        inference: ModelInference,
        left_context: usize,
        right_context: usize,
    ) -> Result<Vec<Prediction>> {
        let old_spkcache_len = self.spkcache.len();
        let old_fifo_len = self.fifo.len();
        let valid_chunk_len = inference.chunk_embedding_length;
        if left_context.saturating_add(right_context) > valid_chunk_len {
            return Err(Error::Diarization(format!(
                "chunk embedding length {valid_chunk_len} is smaller than contexts {left_context}+{right_context}"
            )));
        }
        let main_chunk_len = valid_chunk_len - left_context - right_context;
        let total_state_len = old_spkcache_len
            .saturating_add(old_fifo_len)
            .saturating_add(valid_chunk_len);
        if inference.predictions.len() < total_state_len {
            return Err(Error::Diarization(format!(
                "Sortformer returned {} predictions for {total_state_len} valid state frames",
                inference.predictions.len()
            )));
        }
        if inference.chunk_embeddings.len() < valid_chunk_len {
            return Err(Error::Diarization(format!(
                "Sortformer returned {} embeddings for {valid_chunk_len} valid chunk frames",
                inference.chunk_embeddings.len()
            )));
        }

        self.fifo_preds =
            inference.predictions[old_spkcache_len..old_spkcache_len + old_fifo_len].to_vec();
        let prediction_start = old_spkcache_len + old_fifo_len + left_context;
        let chunk_predictions =
            inference.predictions[prediction_start..prediction_start + main_chunk_len].to_vec();
        let chunk_embeddings =
            inference.chunk_embeddings[left_context..left_context + main_chunk_len].to_vec();

        self.fifo.extend(chunk_embeddings);
        self.fifo_preds.extend_from_slice(&chunk_predictions);
        if self.fifo.len() > FIFO_LEN {
            let overflow = self.fifo.len() - FIFO_LEN;
            let pop_count = UPDATE_PERIOD.max(overflow).min(self.fifo.len());
            let popped_embeddings = self.fifo.drain(..pop_count).collect::<Vec<_>>();
            let popped_predictions = self.fifo_preds.drain(..pop_count).collect::<Vec<_>>();
            self.update_silence_profile(&popped_embeddings, &popped_predictions);

            if self.spkcache_preds.is_none()
                && self.spkcache.len().saturating_add(pop_count) > SPEAKER_CACHE_LEN
            {
                self.spkcache_preds = Some(
                    inference.predictions[..old_spkcache_len]
                        .iter()
                        .copied()
                        .chain(popped_predictions.iter().copied())
                        .collect(),
                );
            } else if let Some(predictions) = self.spkcache_preds.as_mut() {
                predictions.extend_from_slice(&popped_predictions);
            }
            self.spkcache.extend(popped_embeddings);

            if self.spkcache.len() > SPEAKER_CACHE_LEN {
                let predictions = self.spkcache_preds.take().ok_or_else(|| {
                    Error::Diarization(
                        "speaker cache has no predictions during compression".to_string(),
                    )
                })?;
                let (embeddings, predictions) = compress_speaker_cache(
                    &self.spkcache,
                    &predictions,
                    &self.mean_silence_embedding,
                )?;
                self.spkcache = embeddings;
                self.spkcache_preds = Some(predictions);
            }
        }
        Ok(chunk_predictions)
    }

    fn update_silence_profile(&mut self, embeddings: &[Embedding], predictions: &[Prediction]) {
        let mut new_sum = [0.0f32; EMBEDDING_SIZE];
        let mut new_count = 0u64;
        for (embedding, prediction) in embeddings.iter().zip(predictions) {
            if prediction.iter().sum::<f32>() < SILENCE_THRESHOLD {
                for (sum, value) in new_sum.iter_mut().zip(embedding) {
                    *sum += *value;
                }
                new_count = new_count.saturating_add(1);
            }
        }
        if new_count == 0 {
            return;
        }
        let total_count = self.silence_frames.saturating_add(new_count);
        for (mean, sum) in self.mean_silence_embedding.iter_mut().zip(new_sum) {
            let previous_sum = *mean * self.silence_frames as f32;
            *mean = (previous_sum + sum) / total_count as f32;
        }
        self.silence_frames = total_count;
    }
}

fn compress_speaker_cache(
    embeddings: &[Embedding],
    predictions: &[Prediction],
    mean_silence_embedding: &Embedding,
) -> Result<(Vec<Embedding>, Vec<Prediction>)> {
    if embeddings.len() != predictions.len() || embeddings.len() <= SPEAKER_CACHE_LEN {
        return Err(Error::Diarization(format!(
            "invalid speaker cache compression input: {} embeddings, {} predictions",
            embeddings.len(),
            predictions.len()
        )));
    }
    let frame_count = embeddings.len();
    let cache_per_speaker = SPEAKER_CACHE_LEN / MAX_SPEAKERS - SILENCE_FRAMES_PER_SPEAKER;
    let strong_count = (cache_per_speaker as f32 * STRONG_BOOST_RATE).floor() as usize;
    let weak_count = (cache_per_speaker as f32 * WEAK_BOOST_RATE).floor() as usize;
    let minimum_positive = (cache_per_speaker as f32 * MIN_POS_SCORES_RATE).floor() as usize;

    let mut scores = vec![[f32::NEG_INFINITY; MAX_SPEAKERS]; frame_count];
    for (frame, prediction) in predictions.iter().enumerate() {
        let inactive_log_sum = prediction
            .iter()
            .map(|probability| (1.0 - probability).max(PRED_SCORE_THRESHOLD).ln())
            .sum::<f32>();
        for speaker in 0..MAX_SPEAKERS {
            let probability = prediction[speaker];
            if probability > 0.5 {
                scores[frame][speaker] = probability.max(PRED_SCORE_THRESHOLD).ln()
                    - (1.0 - probability).max(PRED_SCORE_THRESHOLD).ln()
                    + inactive_log_sum
                    - 0.5f32.ln();
            }
        }
    }

    for speaker in 0..MAX_SPEAKERS {
        let positive_count = scores.iter().filter(|score| score[speaker] > 0.0).count();
        if positive_count >= minimum_positive {
            for score in &mut scores {
                if score[speaker] <= 0.0 {
                    score[speaker] = f32::NEG_INFINITY;
                }
            }
        }
        for score in scores.iter_mut().skip(SPEAKER_CACHE_LEN) {
            if score[speaker].is_finite() {
                score[speaker] += SCORES_BOOST_LATEST;
            }
        }
        boost_top_scores(&mut scores, speaker, strong_count, 2.0);
        boost_top_scores(&mut scores, speaker, weak_count, 1.0);
    }

    let padded_frame_count = frame_count + SILENCE_FRAMES_PER_SPEAKER;
    let mut flattened = Vec::with_capacity(padded_frame_count * MAX_SPEAKERS);
    for speaker in 0..MAX_SPEAKERS {
        for (frame, score) in scores.iter().enumerate() {
            flattened.push((score[speaker], speaker * padded_frame_count + frame));
        }
        for frame in frame_count..padded_frame_count {
            flattened.push((f32::INFINITY, speaker * padded_frame_count + frame));
        }
    }
    flattened.sort_unstable_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    flattened.truncate(SPEAKER_CACHE_LEN);

    const DISABLED_INDEX: usize = 99_999;
    let mut selected = flattened
        .into_iter()
        .map(|(score, flattened_index)| {
            let frame = flattened_index % padded_frame_count;
            let invalid_score = score == f32::NEG_INFINITY;
            let disabled = invalid_score || frame >= frame_count;
            let sort_index = if invalid_score {
                DISABLED_INDEX
            } else {
                flattened_index
            };
            (sort_index, frame, disabled)
        })
        .collect::<Vec<_>>();
    selected.sort_unstable_by_key(|(sort_index, _, _)| *sort_index);

    let mut compressed_embeddings = Vec::with_capacity(SPEAKER_CACHE_LEN);
    let mut compressed_predictions = Vec::with_capacity(SPEAKER_CACHE_LEN);
    for (_, frame, disabled) in selected {
        if disabled {
            compressed_embeddings.push(*mean_silence_embedding);
            compressed_predictions.push([0.0; MAX_SPEAKERS]);
        } else {
            compressed_embeddings.push(embeddings[frame]);
            compressed_predictions.push(predictions[frame]);
        }
    }
    Ok((compressed_embeddings, compressed_predictions))
}

fn boost_top_scores(scores: &mut [[f32; MAX_SPEAKERS]], speaker: usize, count: usize, scale: f32) {
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        scores[*right][speaker]
            .partial_cmp(&scores[*left][speaker])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    let boost = -scale * 0.5f32.ln();
    for index in indices.into_iter().take(count) {
        scores[index][speaker] += boost;
    }
}

struct Frontend {
    hop_length: usize,
    win_length: usize,
    preemphasis: f32,
    log_zero_guard: f32,
    log_zero_guard_add: bool,
    mag_power: f32,
    window: Vec<f32>,
    mel_basis: Array2<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
}

impl Frontend {
    fn from_metadata(session: &Session) -> Result<Self> {
        let win_length = parse_metadata::<usize>(session, "yamabiko.features.win_length")?;
        let hop_length = parse_metadata::<usize>(session, "yamabiko.features.hop_length")?;
        let n_fft = parse_metadata::<usize>(session, "yamabiko.features.n_fft")?;
        let frame_splicing = parse_metadata::<usize>(session, "yamabiko.features.frame_splicing")?;
        let preemphasis = parse_metadata::<f32>(session, "yamabiko.features.preemph")?;
        let log_zero_guard_value =
            required_metadata(session, "yamabiko.features.log_zero_guard_value")?;
        let log_zero_guard =
            parse_metadata_value::<f32>(log_zero_guard_value.clone()).or_else(|_| {
                match log_zero_guard_value.as_str() {
                    "tiny" => Ok(f32::MIN_POSITIVE),
                    "eps" => Ok(f32::EPSILON),
                    value => Err(Error::DiarizationModelLoad(format!(
                        "unsupported yamabiko.features.log_zero_guard_value '{value}'"
                    ))),
                }
            })?;
        let mag_power = parse_metadata::<f32>(session, "yamabiko.features.mag_power")?;
        let low_frequency = parse_metadata::<f64>(session, "yamabiko.features.lowfreq")?;
        let high_frequency = optional_metadata(session, "yamabiko.features.highfreq")
            .map(parse_metadata_value::<f64>)
            .transpose()?
            .unwrap_or(PCM_SAMPLE_RATE_HZ as f64 / 2.0);

        require_metadata_eq(session, "yamabiko.features.window", "hann")?;
        require_metadata_eq(session, "yamabiko.features.normalize", "NA")?;
        require_metadata_eq(session, "yamabiko.features.mel_norm", "slaney")?;
        require_metadata_eq(session, "yamabiko.features.log", "true")?;
        let log_zero_guard_add =
            match required_metadata(session, "yamabiko.features.log_zero_guard_type")?.as_str() {
                "add" => true,
                "clamp" => false,
                value => {
                    return Err(Error::DiarizationModelLoad(format!(
                        "unsupported yamabiko.features.log_zero_guard_type '{value}'"
                    )));
                }
            };
        require_metadata_eq(session, "yamabiko.features.exact_pad", "false")?;
        let pad_value = parse_metadata::<f32>(session, "yamabiko.features.pad_value")?;
        let _pad_to = parse_metadata::<usize>(session, "yamabiko.features.pad_to")?;
        let dither = parse_metadata::<f32>(session, "yamabiko.features.dither")?;
        if required_metadata(session, "yamabiko.features.config")?.is_empty() {
            return Err(Error::DiarizationModelLoad(
                "yamabiko.features.config must not be empty".to_string(),
            ));
        }

        if hop_length != 160 {
            return Err(Error::DiarizationModelLoad(format!(
                "Sortformer hop length must be 160 samples, got {hop_length}"
            )));
        }
        if win_length != DEFAULT_WIN_LENGTH || win_length > n_fft {
            return Err(Error::DiarizationModelLoad(format!(
                "Sortformer window length must be {DEFAULT_WIN_LENGTH} samples and no larger than n_fft, got win_length={win_length}, n_fft={n_fft}"
            )));
        }
        if frame_splicing != 1 {
            return Err(Error::DiarizationModelLoad(format!(
                "frame splicing {frame_splicing} is unsupported; expected 1"
            )));
        }
        if !preemphasis.is_finite() || !(0.0..=1.0).contains(&preemphasis) {
            return Err(Error::DiarizationModelLoad(format!(
                "invalid preemphasis coefficient {preemphasis}"
            )));
        }
        if !log_zero_guard.is_finite() || log_zero_guard <= 0.0 {
            return Err(Error::DiarizationModelLoad(format!(
                "invalid log zero guard {log_zero_guard}"
            )));
        }
        if mag_power != 1.0 && mag_power != 2.0 {
            return Err(Error::DiarizationModelLoad(format!(
                "unsupported magnitude power {mag_power}; expected 1 or 2"
            )));
        }
        if pad_value != 0.0 {
            return Err(Error::DiarizationModelLoad(format!(
                "unsupported feature pad value {pad_value}; expected 0"
            )));
        }
        if !dither.is_finite() || dither < 0.0 {
            return Err(Error::DiarizationModelLoad(format!(
                "invalid feature dither value {dither}"
            )));
        }
        if !(0.0..high_frequency).contains(&low_frequency)
            || high_frequency > PCM_SAMPLE_RATE_HZ as f64 / 2.0
        {
            return Err(Error::DiarizationModelLoad(format!(
                "invalid Mel frequency range {low_frequency}..{high_frequency} Hz"
            )));
        }

        let window = hann_window(win_length);
        let mel_basis = slaney_mel_filterbank(
            n_fft,
            FEATURE_SIZE,
            PCM_SAMPLE_RATE_HZ as usize,
            low_frequency,
            high_frequency,
        );
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(n_fft);
        Ok(Self {
            hop_length,
            win_length,
            preemphasis,
            log_zero_guard,
            log_zero_guard_add,
            mag_power,
            window,
            mel_basis,
            fft_plan,
        })
    }
}

struct StreamingFeatures {
    frontend: Arc<Frontend>,
    stft_workspace: StftWorkspace,
    raw: VecDeque<f32>,
    raw_start: u64,
    received: u64,
    next_feature: u64,
    feature_start: u64,
    frames: VecDeque<[f32; FEATURE_SIZE]>,
    finished: bool,
}

impl StreamingFeatures {
    fn new(frontend: Arc<Frontend>) -> Self {
        let stft_workspace = StftWorkspace::new(frontend.fft_plan.as_ref());
        Self {
            frontend,
            stft_workspace,
            raw: VecDeque::new(),
            raw_start: 0,
            received: 0,
            next_feature: 0,
            feature_start: 0,
            frames: VecDeque::new(),
            finished: false,
        }
    }

    fn received_samples(&self) -> u64 {
        self.received
    }

    fn available_end(&self) -> u64 {
        self.feature_start.saturating_add(self.frames.len() as u64)
    }

    fn push(&mut self, samples: &[f32]) -> Result<()> {
        if self.finished {
            return Err(Error::Diarization(
                "cannot add samples to a finished feature stream".to_string(),
            ));
        }
        self.received = self
            .received
            .checked_add(samples.len() as u64)
            .ok_or_else(|| Error::Diarization("audio sample timeline overflow".to_string()))?;
        self.raw.extend(samples.iter().copied());
        self.generate(false)
    }

    fn finish(&mut self) -> Result<()> {
        self.finished = true;
        self.generate(true)
    }

    fn generate(&mut self, pad_right: bool) -> Result<()> {
        let valid_features = self.received / self.frontend.hop_length as u64;
        while self.next_feature < valid_features {
            let center = self
                .next_feature
                .checked_mul(self.frontend.hop_length as u64)
                .ok_or_else(|| Error::Diarization("feature timeline overflow".to_string()))?;
            let right_extent = (self.frontend.win_length - self.frontend.win_length / 2 - 1) as u64;
            if !pad_right && center.saturating_add(right_extent) >= self.received {
                break;
            }
            let frame = extract_feature_frame(
                self.frontend.as_ref(),
                &mut self.stft_workspace,
                &self.raw,
                self.raw_start,
                self.received,
                center,
            )?;
            self.frames.push_back(frame);
            self.next_feature = self.next_feature.saturating_add(1);
        }
        self.prune_raw();
        Ok(())
    }

    fn prune_raw(&mut self) {
        let next_center = self
            .next_feature
            .saturating_mul(self.frontend.hop_length as u64);
        let keep_from = next_center
            .saturating_sub(self.frontend.win_length as u64 / 2)
            .saturating_sub(1);
        let discard = keep_from
            .saturating_sub(self.raw_start)
            .min(self.raw.len() as u64);
        self.raw.drain(..discard as usize);
        self.raw_start = self.raw_start.saturating_add(discard);
    }

    fn copy_range(&self, start: u64, end: u64) -> Result<Vec<f32>> {
        if start > end || start < self.feature_start || end > self.available_end() {
            return Err(Error::Diarization(format!(
                "feature range {start}..{end} is unavailable; retained range is {}..{}",
                self.feature_start,
                self.available_end()
            )));
        }
        let offset = usize::try_from(start - self.feature_start)
            .map_err(|_| Error::Diarization("feature offset exceeds usize".to_string()))?;
        let count = usize::try_from(end - start)
            .map_err(|_| Error::Diarization("feature length exceeds usize".to_string()))?;
        let mut output = Vec::with_capacity(count.saturating_mul(FEATURE_SIZE));
        for frame in self.frames.iter().skip(offset).take(count) {
            output.extend_from_slice(frame);
        }
        Ok(output)
    }

    fn discard_before(&mut self, frame: u64) {
        let target = frame.min(self.available_end());
        let count = target
            .saturating_sub(self.feature_start)
            .min(self.frames.len() as u64);
        self.frames.drain(..count as usize);
        self.feature_start = self.feature_start.saturating_add(count);
    }

    #[cfg(test)]
    fn retained_raw_samples(&self) -> usize {
        self.raw.len()
    }
}

fn extract_feature_frame(
    frontend: &Frontend,
    workspace: &mut StftWorkspace,
    raw: &VecDeque<f32>,
    raw_start: u64,
    received: u64,
    center: u64,
) -> Result<[f32; FEATURE_SIZE]> {
    let fft_output = workspace
        .execute_centered(
            frontend.fft_plan.as_ref(),
            &frontend.window,
            center as i128,
            |sample_index| {
                preemphasized_sample(raw, raw_start, received, sample_index, frontend.preemphasis)
            },
        )
        .map_err(|err| Error::Diarization(format!("feature FFT failed: {err}")))?;

    let mut feature = [0.0f32; FEATURE_SIZE];
    for (mel, output) in feature.iter_mut().enumerate() {
        let mut energy = 0.0f32;
        for (frequency, value) in fft_output.iter().enumerate() {
            let magnitude = if frontend.mag_power == 2.0 {
                value.norm_sqr()
            } else {
                value.norm()
            };
            energy += frontend.mel_basis[[mel, frequency]] * magnitude;
        }
        *output = if frontend.log_zero_guard_add {
            (energy + frontend.log_zero_guard).ln()
        } else {
            energy.max(frontend.log_zero_guard).ln()
        };
    }
    Ok(feature)
}

fn preemphasized_sample(
    raw: &VecDeque<f32>,
    raw_start: u64,
    received: u64,
    sample_index: i128,
    coefficient: f32,
) -> f32 {
    if sample_index < 0 || sample_index >= received as i128 {
        return 0.0;
    }
    let index = sample_index as u64;
    let current = retained_sample(raw, raw_start, index).unwrap_or(0.0);
    if index == 0 {
        current
    } else {
        current - coefficient * retained_sample(raw, raw_start, index - 1).unwrap_or(0.0)
    }
}

fn retained_sample(raw: &VecDeque<f32>, raw_start: u64, index: u64) -> Option<f32> {
    let offset = usize::try_from(index.checked_sub(raw_start)?).ok()?;
    raw.get(offset).copied()
}

fn validate_model_contract(session: &Session, precision: ModelPrecision) -> Result<()> {
    let required_values = [
        ("yamabiko.model.kind", MODEL_KIND),
        ("yamabiko.model.id", MODEL_ID),
        ("yamabiko.model.revision", MODEL_REVISION),
        ("yamabiko.export.nemo_version", NEMO_VERSION),
        ("yamabiko.contract.version", CONTRACT_VERSION),
        ("yamabiko.model.precision", precision.metadata_value()),
        ("yamabiko.precision.internal", precision.metadata_value()),
        ("yamabiko.precision.external_io", "float32"),
        ("yamabiko.precision.integer_quantization", "false"),
        ("yamabiko.model.embedding_dim", "512"),
        ("yamabiko.diarization.max_speakers", "4"),
        ("yamabiko.diarization.frame_ms", "80"),
        ("yamabiko.diarization.onset", "0.5"),
        ("yamabiko.diarization.offset", "0.5"),
        ("yamabiko.diarization.pad_onset", "0.0"),
        ("yamabiko.diarization.pad_offset", "0.0"),
        ("yamabiko.diarization.min_duration_on", "0.0"),
        ("yamabiko.diarization.min_duration_off", "0.0"),
        ("yamabiko.streaming.preset", "low_latency"),
        ("yamabiko.streaming.latency_ms", "1040"),
        ("yamabiko.streaming.chunk_len", "6"),
        ("yamabiko.streaming.right_context", "7"),
        ("yamabiko.streaming.left_context", "1"),
        ("yamabiko.streaming.fifo_len", "188"),
        ("yamabiko.streaming.update_period", "144"),
        ("yamabiko.streaming.speaker_cache", "188"),
        ("yamabiko.streaming.state_update", "synchronous"),
        ("yamabiko.streaming.input_buffer_ms", "1040"),
        ("yamabiko.features.subsampling_factor", "8"),
        ("yamabiko.audio.sample_rate", "16000"),
        ("yamabiko.features.n_mels", "128"),
        (
            "yamabiko.tensor.inputs",
            "chunk,chunk_lengths,spkcache,spkcache_lengths,fifo,fifo_lengths",
        ),
        (
            "yamabiko.tensor.outputs",
            "spkcache_fifo_chunk_preds,chunk_pre_encode_embs,chunk_pre_encode_lengths",
        ),
        ("yamabiko.tensor.input.chunk", "float32;rank=3"),
        ("yamabiko.tensor.input.chunk_lengths", "int64;rank=1"),
        ("yamabiko.tensor.input.spkcache", "float32;rank=3"),
        ("yamabiko.tensor.input.spkcache_lengths", "int64;rank=1"),
        ("yamabiko.tensor.input.fifo", "float32;rank=3"),
        ("yamabiko.tensor.input.fifo_lengths", "int64;rank=1"),
        (
            "yamabiko.tensor.output.spkcache_fifo_chunk_preds",
            "float32;rank=3",
        ),
        (
            "yamabiko.tensor.output.chunk_pre_encode_embs",
            "float32;rank=3",
        ),
        (
            "yamabiko.tensor.output.chunk_pre_encode_lengths",
            "int64;rank=1",
        ),
    ];
    for (key, expected) in required_values {
        require_metadata_eq(session, key, expected)?;
    }
    require_metadata_value_eq(
        session,
        "yamabiko.streaming.speaker_cache_silence_frames_per_speaker",
        SILENCE_FRAMES_PER_SPEAKER,
    )?;
    for (key, expected) in [
        ("yamabiko.streaming.silence_threshold", SILENCE_THRESHOLD),
        (
            "yamabiko.streaming.prediction_score_threshold",
            PRED_SCORE_THRESHOLD,
        ),
        (
            "yamabiko.streaming.scores_boost_latest",
            SCORES_BOOST_LATEST,
        ),
        ("yamabiko.streaming.strong_boost_rate", STRONG_BOOST_RATE),
        ("yamabiko.streaming.weak_boost_rate", WEAK_BOOST_RATE),
        (
            "yamabiko.streaming.minimum_positive_scores_rate",
            MIN_POS_SCORES_RATE,
        ),
    ] {
        require_metadata_value_eq(session, key, expected)?;
    }

    require_outlet_set(
        session.inputs(),
        "Sortformer input",
        &[
            "chunk",
            "chunk_lengths",
            "spkcache",
            "spkcache_lengths",
            "fifo",
            "fifo_lengths",
        ],
    )?;
    require_outlet_set(
        session.outputs(),
        "Sortformer output",
        &[
            "spkcache_fifo_chunk_preds",
            "chunk_pre_encode_embs",
            "chunk_pre_encode_lengths",
        ],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "chunk",
        TensorElementType::Float32,
        3,
        &[(0, 1), (2, FEATURE_SIZE)],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "chunk_lengths",
        TensorElementType::Int64,
        1,
        &[(0, 1)],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "spkcache",
        TensorElementType::Float32,
        3,
        &[(0, 1), (2, EMBEDDING_SIZE)],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "spkcache_lengths",
        TensorElementType::Int64,
        1,
        &[(0, 1)],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "fifo",
        TensorElementType::Float32,
        3,
        &[(0, 1), (2, EMBEDDING_SIZE)],
    )?;
    require_tensor(
        session.inputs(),
        "Sortformer input",
        "fifo_lengths",
        TensorElementType::Int64,
        1,
        &[(0, 1)],
    )?;
    require_tensor(
        session.outputs(),
        "Sortformer output",
        "spkcache_fifo_chunk_preds",
        TensorElementType::Float32,
        3,
        &[(0, 1), (2, MAX_SPEAKERS)],
    )?;
    require_tensor(
        session.outputs(),
        "Sortformer output",
        "chunk_pre_encode_embs",
        TensorElementType::Float32,
        3,
        &[(0, 1), (2, EMBEDDING_SIZE)],
    )?;
    require_tensor(
        session.outputs(),
        "Sortformer output",
        "chunk_pre_encode_lengths",
        TensorElementType::Int64,
        1,
        &[(0, 1)],
    )?;
    Ok(())
}

fn require_outlet_set(
    outlets: &[ort::value::Outlet],
    location: &str,
    expected: &[&str],
) -> Result<()> {
    ort_utils::require_outlet_set(outlets, location, expected).map_err(Error::DiarizationModelLoad)
}

fn require_tensor(
    outlets: &[ort::value::Outlet],
    location: &str,
    name: &str,
    expected_type: TensorElementType,
    expected_rank: usize,
    dimensions: &[(usize, usize)],
) -> Result<()> {
    ort_utils::require_tensor(
        outlets,
        location,
        name,
        expected_type,
        Some(expected_rank),
        dimensions,
    )
    .map(|_| ())
    .map_err(Error::DiarizationModelLoad)
}

fn required_metadata(session: &Session, key: &str) -> Result<String> {
    session
        .metadata()
        .map_err(|err| Error::DiarizationModelLoad(err.to_string()))?
        .custom(key)
        .ok_or_else(|| Error::DiarizationModelLoad(format!("model metadata is missing '{key}'")))
}

fn optional_metadata(session: &Session, key: &str) -> Option<String> {
    session.metadata().ok()?.custom(key)
}

fn require_metadata_eq(session: &Session, key: &str, expected: &str) -> Result<()> {
    let actual = required_metadata(session, key)?;
    if actual != expected {
        return Err(Error::DiarizationModelLoad(format!(
            "model metadata '{key}' must be '{expected}', got '{actual}'"
        )));
    }
    Ok(())
}

fn require_metadata_value_eq<T>(session: &Session, key: &str, expected: T) -> Result<()>
where
    T: FromStr + PartialEq + std::fmt::Display,
    T::Err: std::fmt::Display,
{
    let actual = parse_metadata::<T>(session, key)?;
    if actual != expected {
        return Err(Error::DiarizationModelLoad(format!(
            "model metadata '{key}' must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn parse_metadata<T>(session: &Session, key: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    parse_metadata_value(required_metadata(session, key)?)
}

fn parse_metadata_value<T>(value: String) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    value.parse::<T>().map_err(|err| {
        Error::DiarizationModelLoad(format!("invalid metadata value '{value}': {err}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frontend() -> Arc<Frontend> {
        let n_fft = 512;
        let win_length = DEFAULT_WIN_LENGTH;
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        Arc::new(Frontend {
            hop_length: 160,
            win_length,
            preemphasis: 0.97,
            log_zero_guard: 2.0f32.powi(-24),
            log_zero_guard_add: true,
            mag_power: 2.0,
            window: hann_window(win_length),
            mel_basis: slaney_mel_filterbank(n_fft, FEATURE_SIZE, 16_000, 0.0, 8_000.0),
            fft_plan: planner.plan_fft_forward(n_fft),
        })
    }

    fn test_audio(sample_count: usize) -> Vec<f32> {
        (0..sample_count)
            .map(|index| {
                ((index as f32) * 0.017).sin() * 0.2 + ((index as f32) * 0.011).cos() * 0.05
            })
            .collect()
    }

    fn extract_features(partitions: &[usize]) -> Vec<f32> {
        let audio = test_audio(partitions.iter().sum());
        let mut stream = StreamingFeatures::new(test_frontend());
        let mut offset = 0;
        for &length in partitions {
            stream.push(&audio[offset..offset + length]).unwrap();
            offset += length;
        }
        stream.finish().unwrap();
        stream.copy_range(0, stream.available_end()).unwrap()
    }

    #[test]
    fn provider_candidates_enforce_precision_and_priority() {
        assert_eq!(
            model_candidates(Device::Cpu).unwrap(),
            [ModelCandidate {
                provider: ProviderKind::Cpu,
                precision: ModelPrecision::Fp32,
            }]
        );

        let expected = vec![
            #[cfg(feature = "cuda")]
            ModelCandidate {
                provider: ProviderKind::Cuda,
                precision: ModelPrecision::Fp16,
            },
            #[cfg(feature = "directml")]
            ModelCandidate {
                provider: ProviderKind::DirectMl,
                precision: ModelPrecision::Fp16,
            },
            ModelCandidate {
                provider: ProviderKind::Cpu,
                precision: ModelPrecision::Fp32,
            },
        ];
        assert_eq!(model_candidates(Device::Auto).unwrap(), expected);
    }

    #[test]
    fn unsupported_provider_is_rejected_before_model_path_validation() {
        let error = SortformerDiarizer::load(
            Path::new("directory-that-does-not-exist"),
            Device::TensorRt,
            1,
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            Error::DeviceUnavailable {
                device: Device::TensorRt,
                ..
            }
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn explicit_cuda_requires_the_cargo_feature() {
        assert!(matches!(
            model_candidates(Device::Cuda),
            Err(Error::DeviceUnavailable {
                device: Device::Cuda,
                ..
            })
        ));
    }

    #[cfg(not(feature = "directml"))]
    #[test]
    fn explicit_directml_requires_the_cargo_feature() {
        assert!(matches!(
            model_candidates(Device::DirectMl),
            Err(Error::DeviceUnavailable {
                device: Device::DirectMl,
                ..
            })
        ));
    }

    #[test]
    fn model_filenames_match_precision_contract() {
        assert_eq!(ModelPrecision::Fp32.filename(), "sortformer.fp32.onnx");
        assert_eq!(ModelPrecision::Fp16.filename(), "sortformer.fp16.onnx");
    }

    #[test]
    fn retained_pcm_limit_is_lookahead_times_max_sources() {
        assert_eq!(MODEL_LOOKAHEAD_SAMPLES, 16_840);
        assert_eq!(retained_pcm_limit(3).unwrap(), 50_520);
        assert!(retained_pcm_limit(0).is_err());
        assert!(retained_pcm_limit(usize::MAX).is_err());
    }

    #[test]
    fn final_window_pads_only_right_context() {
        assert_eq!(padded_window_frames(0, MAIN_FEATURE_FRAMES), 104);
        assert_eq!(padded_window_frames(LEFT_FEATURE_FRAMES, 17), 81);
        assert_eq!(padded_window_frames(LEFT_FEATURE_FRAMES, 1), 65);
    }

    #[test]
    fn final_partial_window_returns_only_real_main_predictions() {
        let predictions = (0..11)
            .map(|frame| [frame as f32; MAX_SPEAKERS])
            .collect::<Vec<_>>();
        let inference = ModelInference {
            predictions: predictions.clone(),
            chunk_embeddings: vec![[0.0; EMBEDDING_SIZE]; 11],
            chunk_embedding_length: 11,
        };
        let mut cache = StreamingCache::default();

        let main = cache.update(inference, 1, RIGHT_CONTEXT).unwrap();

        assert_eq!(main, predictions[1..4]);
        assert_eq!(cache.fifo.len(), 3);
    }

    #[test]
    fn empty_state_uses_a_nonzero_physical_tensor() {
        let state = embeddings_to_array(&[]).unwrap();
        assert_eq!(state.shape(), [1, 1, EMBEDDING_SIZE]);
        assert!(state.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn adjacent_regions_with_the_same_activity_are_merged() {
        let mut regions = Vec::new();
        append_region(&mut regions, 0, 10, [true, false, false, false]);
        append_region(&mut regions, 10, 20, [true, false, false, false]);
        assert_eq!(
            regions,
            vec![DiarizedRegion {
                start_sample: 0,
                end_sample: 20,
                speakers: vec![BackendSpeakerId::new(0)],
            }]
        );
    }

    #[test]
    fn overlapping_activity_has_multiple_speakers_in_one_region() {
        let mut regions = Vec::new();
        append_region(&mut regions, 0, 1_280, [true, true, false, false]);
        assert_eq!(
            regions[0].speakers,
            [BackendSpeakerId::new(0), BackendSpeakerId::new(1)]
        );
    }

    #[test]
    fn cache_compression_matches_nemo_2_7_3_golden_selection() {
        let embeddings = (0..SPEAKER_CACHE_LEN + 10)
            .map(|frame| {
                let mut embedding = [0.0; EMBEDDING_SIZE];
                embedding[0] = frame as f32;
                embedding
            })
            .collect::<Vec<_>>();
        let predictions = (0..embeddings.len())
            .map(|frame| {
                let mut prediction = [0.0; MAX_SPEAKERS];
                prediction[frame % MAX_SPEAKERS] = 0.55 + frame as f32 / 1_000.0;
                prediction
            })
            .collect::<Vec<_>>();
        let mut silence_embedding = [0.0; EMBEDDING_SIZE];
        silence_embedding[0] = -7.0;
        let (compressed_embeddings, compressed_predictions) =
            compress_speaker_cache(&embeddings, &predictions, &silence_embedding).unwrap();
        assert_eq!(compressed_embeddings.len(), SPEAKER_CACHE_LEN);
        assert_eq!(compressed_predictions.len(), SPEAKER_CACHE_LEN);
        let silence_positions = compressed_embeddings
            .iter()
            .enumerate()
            .filter_map(|(position, embedding)| (embedding[0] == -7.0).then_some(position))
            .collect::<Vec<_>>();
        assert_eq!(
            silence_positions,
            [44, 45, 46, 91, 92, 93, 138, 139, 140, 185, 186, 187]
        );
        let checksum = compressed_embeddings
            .iter()
            .enumerate()
            .map(|(position, embedding)| (position as i64 + 1) * (embedding[0] as i64 + 8))
            .sum::<i64>();
        assert_eq!(checksum, 2_031_954);
        let prediction_sums = (0..MAX_SPEAKERS)
            .map(|speaker| {
                compressed_predictions
                    .iter()
                    .map(|prediction| prediction[speaker])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (actual, expected) in prediction_sums
            .into_iter()
            .zip([29.04, 29.084, 28.952, 28.996])
        {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn frontend_matches_nemo_2_7_3_golden_values() {
        let features = extract_features(&[1_600]);
        let golden = [
            (0, 0, -4.5632954f32),
            (0, 1, -4.1383266),
            (0, 9, -4.205631),
            (1, 0, -4.7884717),
            (10, 3, -16.174662),
            (40, 5, -16.635433),
            (79, 9, -16.519491),
            (100, 4, -16.635532),
            (127, 9, -16.60932),
        ];
        for (mel, frame, expected) in golden {
            let actual = features[frame * FEATURE_SIZE + mel];
            assert!(
                (actual - expected).abs() < 3.0e-4,
                "mel={mel}, frame={frame}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn frontend_is_invariant_to_arbitrary_push_partitions() {
        let contiguous = extract_features(&[4_173]);
        let partitioned = extract_features(&[1, 159, 17, 801, 2_000, 1_195]);
        assert_eq!(contiguous, partitioned);
    }

    #[test]
    fn first_cache_refresh_matches_nemo_2_7_3_synchronous_update() {
        let embedding = |id: usize| {
            let mut value = [0.0; EMBEDDING_SIZE];
            value[0] = id as f32;
            value
        };
        let mut cache = StreamingCache {
            spkcache: (0..144).map(embedding).collect(),
            spkcache_preds: None,
            fifo: (144..332).map(embedding).collect(),
            fifo_preds: Vec::new(),
            mean_silence_embedding: {
                let mut value = [0.0; EMBEDDING_SIZE];
                value[0] = -7.0;
                value
            },
            silence_frames: 0,
        };
        let mut predictions = vec![[0.0; MAX_SPEAKERS]; 144 + 188 + 13];
        for (index, prediction) in predictions[..144].iter_mut().enumerate() {
            prediction[0] = 0.55 + index as f32 / 1_000.0;
        }
        for (index, prediction) in predictions[144..332].iter_mut().enumerate() {
            prediction[1] = 0.56 + index as f32 / 2_000.0;
            prediction[2] = 0.60 + index as f32 / 2_500.0;
        }
        for (index, prediction) in predictions[332..].iter_mut().enumerate() {
            prediction[3] = 0.70 + index as f32 / 1_000.0;
        }
        let inference = ModelInference {
            predictions,
            chunk_embeddings: (332..345).map(embedding).collect(),
            chunk_embedding_length: 13,
        };

        let main = cache.update(inference, 0, RIGHT_CONTEXT).unwrap();

        assert_eq!(main.len(), CHUNK_LEN);
        assert!((main.iter().map(|prediction| prediction[3]).sum::<f32>() - 4.215).abs() < 1.0e-5);
        assert_eq!(cache.spkcache.len(), SPEAKER_CACHE_LEN);
        assert_eq!(cache.fifo.len(), 50);
        assert_eq!(cache.fifo.first().unwrap()[0], 288.0);
        assert_eq!(cache.fifo.last().unwrap()[0], 337.0);
        let silence_positions = cache
            .spkcache
            .iter()
            .enumerate()
            .filter_map(|(position, embedding)| (embedding[0] == -7.0).then_some(position))
            .collect::<Vec<_>>();
        assert_eq!(
            silence_positions,
            [110, 111, 112, 146, 147, 148, 182, 183, 184, 185, 186, 187]
        );
        let checksum = cache
            .spkcache
            .iter()
            .enumerate()
            .map(|(position, embedding)| (position as i64 + 1) * (embedding[0] as i64 + 8))
            .sum::<i64>();
        assert_eq!(checksum, 2_778_733);
        let cache_predictions = cache.spkcache_preds.unwrap();
        for (speaker, expected) in [70.235, 38.94, 41.184, 0.0].into_iter().enumerate() {
            let actual = cache_predictions
                .iter()
                .map(|prediction| prediction[speaker])
                .sum::<f32>();
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn source_frontends_keep_independent_timelines_and_bounded_raw_pcm() {
        let frontend = test_frontend();
        let mut first = StreamingFeatures::new(Arc::clone(&frontend));
        let mut second = StreamingFeatures::new(frontend);
        let audio = test_audio(80_000);
        for chunk in audio.chunks(317) {
            first.push(chunk).unwrap();
            assert!(first.retained_raw_samples() <= DEFAULT_WIN_LENGTH + 317);
        }
        second.push(&audio[..1_234]).unwrap();
        assert_eq!(first.received_samples(), 80_000);
        assert_eq!(second.received_samples(), 1_234);
        assert_ne!(first.available_end(), second.available_end());
    }

    #[test]
    fn fifo_pop_updates_silence_profile_and_compresses_cache() {
        let mut cache = StreamingCache::default();
        for _ in 0..56 {
            let state_frames = cache.spkcache.len() + cache.fifo.len();
            let inference = ModelInference {
                predictions: vec![[0.0; MAX_SPEAKERS]; state_frames + 13],
                chunk_embeddings: vec![[1.0; EMBEDDING_SIZE]; 13],
                chunk_embedding_length: 13,
            };
            let output = cache.update(inference, 0, RIGHT_CONTEXT).unwrap();
            assert_eq!(output.len(), CHUNK_LEN);
        }
        assert_eq!(cache.spkcache.len(), SPEAKER_CACHE_LEN);
        assert!(cache.spkcache_preds.is_some());
        assert_eq!(cache.fifo.len(), 48);
        assert_eq!(cache.fifo_preds.len(), 48);
        assert_eq!(cache.silence_frames, 288);
        assert!(
            cache
                .mean_silence_embedding
                .iter()
                .all(|value| (*value - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn activity_postprocessing_preserves_overlap_and_threshold_edges() {
        let mut active = [false; MAX_SPEAKERS];
        apply_activity_threshold(&mut active, [0.5, 0.8, 0.1, 0.0]);
        assert_eq!(active, [false, true, false, false]);
        apply_activity_threshold(&mut active, [0.49, 0.5, 0.9, 0.0]);
        assert_eq!(active, [false, true, true, false]);
    }

    #[test]
    #[ignore = "requires YAMABIKO_DIARIZATION_MODEL_DIR with exported Sortformer models"]
    fn exported_sortformer_runs_the_streaming_contract() {
        let model_dir = std::env::var_os("YAMABIKO_DIARIZATION_MODEL_DIR")
            .map(PathBuf::from)
            .expect("YAMABIKO_DIARIZATION_MODEL_DIR must be set");
        let mut diarizer = SortformerDiarizer::load(&model_dir, Device::Cpu, 2).unwrap();
        let source_id = AudioSourceId::PRIMARY;
        diarizer.open_source(source_id).unwrap();
        let audio = test_audio(20_003);
        let cancelled = AtomicBool::new(false);
        let partitions = [1, 159, 2_048, 333, 4_096];
        let mut start = 0usize;
        let mut partition = 0usize;
        let mut finalized = 0u64;

        while start < audio.len() {
            let end = start
                .saturating_add(partitions[partition % partitions.len()])
                .min(audio.len());
            let output = diarizer
                .push(source_id, &audio[start..end], start as u64, &cancelled)
                .unwrap();
            assert!(output.finalized_until >= finalized);
            assert!(output.finalized_until <= end as u64);
            finalized = output.finalized_until;
            start = end;
            partition += 1;
        }

        let output = diarizer.finish(source_id, &cancelled).unwrap();
        assert_eq!(output.finalized_until, audio.len() as u64);
    }
}
