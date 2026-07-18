"""Export NVIDIA Streaming Sortformer diarization models for yamabiko-asr.

The source revision and NeMo version are intentionally pinned. The exporter
creates both runtime precisions while keeping the public ONNX inputs and
outputs in FP32:

  sortformer.fp32.onnx
  sortformer.fp32.onnx.data
  sortformer.fp16.onnx
  sortformer.fp16.onnx.data
  MODEL_SOURCE.txt

Python, NeMo, and the conversion packages are export-time dependencies only.
The generated models are not distributed or downloaded by the crate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parakeet_export import patch_torch_onnx_export


MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "a494724e2261b51d18a6ef403343b1f7025b3b6d"
NEMO_VERSION = "2.7.3"
MIN_TORCH_VERSION = (2, 6)
UPSTREAM_LICENSE = "NVIDIA Open Model License"

FP32_MODEL_FILENAME = "sortformer.fp32.onnx"
FP32_DATA_FILENAME = "sortformer.fp32.onnx.data"
FP16_MODEL_FILENAME = "sortformer.fp16.onnx"
FP16_DATA_FILENAME = "sortformer.fp16.onnx.data"
SOURCE_FILENAME = "MODEL_SOURCE.txt"
OUTPUT_FILENAMES = (
    FP32_MODEL_FILENAME,
    FP32_DATA_FILENAME,
    FP16_MODEL_FILENAME,
    FP16_DATA_FILENAME,
    SOURCE_FILENAME,
)

CONTRACT_VERSION = "1"
SAMPLE_RATE = 16_000
MEL_BINS = 128
EMBEDDING_DIM = 512
MAX_SPEAKERS = 4
DIARIZATION_FRAME_MS = 80
SUBSAMPLING_FACTOR = 8

CHUNK_LEN = 6
RIGHT_CONTEXT = 7
LEFT_CONTEXT = 1
FIFO_LEN = 188
UPDATE_PERIOD = 144
SPEAKER_CACHE_LEN = 188
SPEAKER_CACHE_SILENCE_FRAMES = 3
SILENCE_THRESHOLD = 0.2
PRED_SCORE_THRESHOLD = 0.25
SCORES_BOOST_LATEST = 0.05
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POS_SCORES_RATE = 0.5
ONSET = 0.5
OFFSET = 0.5
LOW_LATENCY_MS = (CHUNK_LEN + RIGHT_CONTEXT) * DIARIZATION_FRAME_MS
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-3

INPUT_CONTRACT = {
    "chunk": ("float32", 3),
    "chunk_lengths": ("int64", 1),
    "spkcache": ("float32", 3),
    "spkcache_lengths": ("int64", 1),
    "fifo": ("float32", 3),
    "fifo_lengths": ("int64", 1),
}
OUTPUT_CONTRACT = {
    "spkcache_fifo_chunk_preds": ("float32", 3),
    "chunk_pre_encode_embs": ("float32", 3),
    "chunk_pre_encode_lengths": ("int64", 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the pinned nvidia/diar_streaming_sortformer_4spk-v2.1 "
            "revision to yamabiko-asr's FP32 and FP16 ONNX layout"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "diar_streaming_sortformer_4spk-v2.1-onnx",
        help="Directory to write the two ONNX models and MODEL_SOURCE.txt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models") / ".hf-cache",
        help="Hugging Face cache used to download the pinned model revision",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("models") / ".export-work-sortformer",
        help="Temporary directory used for raw and staged ONNX files",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep raw and staged export artifacts for debugging",
    )
    return parser.parse_args()


def installed_distribution(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError(
        f"Missing conversion dependency {names[0]}; install "
        "nemo_toolkit[asr]==2.7.3, torch>=2.6, onnx, onnxruntime, "
        "onnxconverter-common, and huggingface-hub."
    )


def major_minor(version: str) -> tuple[int, int]:
    try:
        major, minor = version.split("+", 1)[0].split(".", 2)[:2]
        return int(major), int(minor)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot parse installed package version {version!r}") from error


def require_export_environment() -> dict[str, str]:
    installed_nemo = installed_distribution("nemo_toolkit")
    if installed_nemo != NEMO_VERSION:
        raise RuntimeError(
            f"NeMo {NEMO_VERSION} is required for a reproducible export, "
            f"found {installed_nemo}"
        )

    installed_torch = installed_distribution("torch")
    if major_minor(installed_torch) < MIN_TORCH_VERSION:
        raise RuntimeError(
            "PyTorch 2.6 or newer is required by NeMo 2.7.3; "
            f"found {installed_torch}"
        )

    versions = {
        "nemo_toolkit": installed_nemo,
        "torch": installed_torch,
        "onnx": installed_distribution("onnx"),
        "onnxruntime": installed_distribution("onnxruntime", "onnxruntime-gpu"),
        "onnxconverter_common": installed_distribution("onnxconverter-common"),
        "huggingface_hub": installed_distribution("huggingface-hub"),
    }
    print(
        "Conversion environment: "
        + ", ".join(f"{name}={version}" for name, version in versions.items())
    )
    return versions


def configure_local_caches(cache_dir: Path) -> dict[str, Path]:
    cache_root = cache_dir.parent.resolve()
    defaults = {
        "HF_HOME": cache_dir.resolve(),
        "NEMO_CACHE_DIR": (cache_root / ".nemo-cache").resolve(),
        "MPLCONFIGDIR": (cache_root / ".mpl-cache").resolve(),
        "NUMBA_CACHE_DIR": (cache_root / ".numba-cache").resolve(),
    }
    configured = {"model download cache": cache_dir.resolve()}
    for variable, default in defaults.items():
        value = os.environ.setdefault(variable, str(default))
        configured[f"{variable} cache"] = Path(value).expanduser().resolve()
    return configured


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_output_paths(
    output_dir: Path,
    work_dir: Path,
    cache_directories: dict[str, Path],
    current_directory: Path | None = None,
) -> None:
    cwd = (current_directory or Path.cwd()).resolve()
    filesystem_root = Path(work_dir.anchor)
    if work_dir == filesystem_root or work_dir == cwd or work_dir in cwd.parents:
        raise RuntimeError(
            f"Refusing unsafe --work-dir {work_dir}: it contains the current directory"
        )
    protected_directories = {
        "output directory": output_dir,
        **cache_directories,
    }
    for label, path in protected_directories.items():
        if paths_overlap(work_dir, path):
            raise RuntimeError(
                f"Refusing unsafe --work-dir {work_dir}: it overlaps {label} {path}"
            )


def download_source_model(cache_dir: Path) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
            allow_patterns=["*.nemo"],
        )
    ).resolve()
    resolved_revision = snapshot.name
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            "Hugging Face resolved the pinned revision to an unexpected commit: "
            f"expected {MODEL_REVISION}, found {resolved_revision}"
        )

    archives = sorted(snapshot.rglob("*.nemo"))
    if len(archives) != 1:
        found = ", ".join(str(path.relative_to(snapshot)) for path in archives) or "none"
        raise RuntimeError(
            f"Expected exactly one .nemo archive in the snapshot, found {found}"
        )
    return archives[0], resolved_revision


def load_model(archive: Path):
    import torch
    from nemo.collections.asr.models.sortformer_diar_models import (
        SortformerEncLabelModel,
    )

    model = SortformerEncLabelModel.restore_from(
        restore_path=str(archive),
        map_location=torch.device("cpu"),
    )
    model.eval()
    model.freeze()
    model = model.cpu()

    modules = model.sortformer_modules
    modules.chunk_len = CHUNK_LEN
    modules.chunk_right_context = RIGHT_CONTEXT
    modules.chunk_left_context = LEFT_CONTEXT
    modules.fifo_len = FIFO_LEN
    modules.spkcache_update_period = UPDATE_PERIOD
    modules.spkcache_len = SPEAKER_CACHE_LEN
    modules._check_streaming_parameters()
    model.streaming_mode = True
    return model


def cfg_value(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


def feature_metadata(model) -> dict[str, str]:
    config = model.cfg.preprocessor
    featurizer = model.preprocessor.featurizer
    sample_rate = int(cfg_value(config, "sample_rate"))
    mel_bins = int(cfg_value(config, "features"))
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"Expected a {SAMPLE_RATE} Hz model, found {sample_rate} Hz")
    if mel_bins != MEL_BINS:
        raise RuntimeError(f"Expected {MEL_BINS} Mel bins, found {mel_bins}")

    values = {
        "sample_rate": sample_rate,
        "features": mel_bins,
        "window_size": cfg_value(
            config, "window_size", featurizer.win_length / sample_rate
        ),
        "window_stride": cfg_value(
            config, "window_stride", featurizer.hop_length / sample_rate
        ),
        "win_length": int(featurizer.win_length),
        "hop_length": int(featurizer.hop_length),
        "n_fft": int(featurizer.n_fft),
        "window": cfg_value(config, "window", "hann"),
        "normalize": featurizer.normalize,
        "preemph": featurizer.preemph,
        "dither": featurizer.dither,
        "pad_to": featurizer.pad_to,
        "pad_value": featurizer.pad_value,
        "frame_splicing": featurizer.frame_splicing,
        "mag_power": featurizer.mag_power,
        "mel_norm": cfg_value(config, "mel_norm", "slaney"),
        "log": featurizer.log,
        "log_zero_guard_type": featurizer.log_zero_guard_type,
        "log_zero_guard_value": featurizer.log_zero_guard_value,
        "lowfreq": cfg_value(config, "lowfreq", 0),
        "highfreq": cfg_value(config, "highfreq", sample_rate / 2),
        "exact_pad": featurizer.exact_pad,
    }
    metadata = {
        "yamabiko.features.config": json.dumps(
            values, sort_keys=True, separators=(",", ":"), default=str
        ),
        "yamabiko.features.n_mels": str(mel_bins),
        "yamabiko.audio.sample_rate": str(sample_rate),
    }
    for name, value in values.items():
        if name not in {"sample_rate", "features"} and value is not None:
            metadata[f"yamabiko.features.{name}"] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )
    return metadata


def model_metadata(
    model,
    resolved_revision: str,
    export_environment: dict[str, str],
) -> dict[str, str]:
    modules = model.sortformer_modules
    if int(modules.n_spk) != MAX_SPEAKERS:
        raise RuntimeError(
            f"Expected a {MAX_SPEAKERS}-speaker model, found {int(modules.n_spk)} speakers"
        )
    if int(modules.fc_d_model) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected embedding dimension {EMBEDDING_DIM}, found {int(modules.fc_d_model)}"
        )
    cache_contract = {
        "spkcache_sil_frames_per_spk": SPEAKER_CACHE_SILENCE_FRAMES,
        "sil_threshold": SILENCE_THRESHOLD,
        "pred_score_threshold": PRED_SCORE_THRESHOLD,
        "scores_boost_latest": SCORES_BOOST_LATEST,
        "strong_boost_rate": STRONG_BOOST_RATE,
        "weak_boost_rate": WEAK_BOOST_RATE,
        "min_pos_scores_rate": MIN_POS_SCORES_RATE,
    }
    for name, expected in cache_contract.items():
        actual = getattr(modules, name)
        if actual != expected:
            raise RuntimeError(f"Expected Sortformer {name}={expected}, found {actual}")
    if int(modules.subsampling_factor) != SUBSAMPLING_FACTOR:
        raise RuntimeError(
            f"Expected subsampling factor {SUBSAMPLING_FACTOR}, "
            f"found {int(modules.subsampling_factor)}"
        )

    metadata = {
        "yamabiko.model.kind": "sortformer_streaming_diarization",
        "yamabiko.model.id": MODEL_ID,
        "yamabiko.model.revision": resolved_revision,
        "yamabiko.export.nemo_version": NEMO_VERSION,
        "yamabiko.contract.version": CONTRACT_VERSION,
        "yamabiko.model.embedding_dim": str(EMBEDDING_DIM),
        "yamabiko.diarization.max_speakers": str(MAX_SPEAKERS),
        "yamabiko.diarization.frame_ms": str(DIARIZATION_FRAME_MS),
        "yamabiko.diarization.onset": str(ONSET),
        "yamabiko.diarization.offset": str(OFFSET),
        "yamabiko.diarization.pad_onset": "0.0",
        "yamabiko.diarization.pad_offset": "0.0",
        "yamabiko.diarization.min_duration_on": "0.0",
        "yamabiko.diarization.min_duration_off": "0.0",
        "yamabiko.streaming.preset": "low_latency",
        "yamabiko.streaming.latency_ms": str(LOW_LATENCY_MS),
        "yamabiko.streaming.chunk_len": str(int(modules.chunk_len)),
        "yamabiko.streaming.right_context": str(int(modules.chunk_right_context)),
        "yamabiko.streaming.left_context": str(int(modules.chunk_left_context)),
        "yamabiko.streaming.fifo_len": str(int(modules.fifo_len)),
        "yamabiko.streaming.update_period": str(int(modules.spkcache_update_period)),
        "yamabiko.streaming.speaker_cache": str(int(modules.spkcache_len)),
        "yamabiko.streaming.state_update": "synchronous",
        "yamabiko.streaming.speaker_cache_silence_frames_per_speaker": str(
            SPEAKER_CACHE_SILENCE_FRAMES
        ),
        "yamabiko.streaming.silence_threshold": str(SILENCE_THRESHOLD),
        "yamabiko.streaming.prediction_score_threshold": str(PRED_SCORE_THRESHOLD),
        "yamabiko.streaming.scores_boost_latest": str(SCORES_BOOST_LATEST),
        "yamabiko.streaming.strong_boost_rate": str(STRONG_BOOST_RATE),
        "yamabiko.streaming.weak_boost_rate": str(WEAK_BOOST_RATE),
        "yamabiko.streaming.minimum_positive_scores_rate": str(
            MIN_POS_SCORES_RATE
        ),
        "yamabiko.streaming.input_buffer_ms": str(LOW_LATENCY_MS),
        "yamabiko.features.subsampling_factor": str(SUBSAMPLING_FACTOR),
        "yamabiko.tensor.inputs": ",".join(INPUT_CONTRACT),
        "yamabiko.tensor.outputs": ",".join(OUTPUT_CONTRACT),
    }
    metadata.update(
        {
            f"yamabiko.export.{name}_version": version
            for name, version in export_environment.items()
            if name != "nemo_toolkit"
        }
    )
    metadata.update(
        {
            f"yamabiko.tensor.input.{name}": f"{element_type};rank={rank}"
            for name, (element_type, rank) in INPUT_CONTRACT.items()
        }
    )
    metadata.update(
        {
            f"yamabiko.tensor.output.{name}": f"{element_type};rank={rank}"
            for name, (element_type, rank) in OUTPUT_CONTRACT.items()
        }
    )
    metadata.update(feature_metadata(model))
    return metadata


def precision_metadata(precision: str) -> dict[str, str]:
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported precision metadata {precision!r}")
    return {
        "yamabiko.model.precision": precision,
        "yamabiko.precision.internal": precision,
        "yamabiko.precision.external_io": "float32",
        "yamabiko.precision.integer_quantization": "false",
    }


def export_safe_concat_and_pad(embs, lengths):
    """ONNX-exportable equivalent of NeMo's scripted concat_and_pad."""
    import torch

    total_lengths = torch.stack(lengths).sum(0)
    joined = torch.cat(embs, dim=1)
    positions = torch.arange(total_lengths.max(), device=joined.device)
    positions = positions.unsqueeze(0).expand(joined.shape[0], -1)

    offsets = torch.zeros_like(positions)
    cumulative_lengths = torch.zeros_like(lengths[0]).unsqueeze(1)
    for index in range(len(embs) - 1):
        cumulative_lengths = cumulative_lengths + lengths[index].unsqueeze(1)
        padding = embs[index].shape[1] - lengths[index].unsqueeze(1)
        offsets = offsets + (positions >= cumulative_lengths).to(positions.dtype) * padding

    source_indices = (positions + offsets).clamp(max=joined.shape[1] - 1)
    output = joined.gather(
        1,
        source_indices.unsqueeze(-1).expand(-1, -1, joined.shape[2]),
    )
    valid = positions < total_lengths.unsqueeze(1)
    output = output * valid.unsqueeze(-1).to(joined.dtype)
    return output, total_lengths


def validate_export_safe_concat(original) -> None:
    import torch

    def embeddings(batch_size: int, capacity: int, dimension: int, offset: int):
        values = torch.arange(batch_size * capacity * dimension, dtype=torch.float32)
        return (values + offset).reshape(batch_size, capacity, dimension)

    cases = [
        (
            [
                embeddings(2, 5, 3, 0),
                embeddings(2, 7, 3, 100),
                embeddings(2, 4, 3, 200),
            ],
            [
                torch.tensor([5, 2]),
                torch.tensor([3, 7]),
                torch.tensor([4, 1]),
            ],
        ),
        (
            [
                embeddings(4, 3, 2, 0),
                embeddings(4, 4, 2, 100),
                embeddings(4, 2, 2, 200),
            ],
            [
                torch.tensor([0, 3, 0, 3]),
                torch.tensor([0, 0, 4, 2]),
                torch.tensor([0, 2, 1, 0]),
            ],
        ),
    ]

    with torch.inference_mode():
        for case_index, (embs, lengths) in enumerate(cases):
            expected_output, expected_lengths = original(embs, lengths)
            actual_output, actual_lengths = export_safe_concat_and_pad(embs, lengths)
            if not torch.equal(actual_lengths, expected_lengths) or not torch.equal(
                actual_output, expected_output
            ):
                raise RuntimeError(
                    "Export-safe concat_and_pad did not exactly match NeMo "
                    f"for self-check case {case_index}"
                )


@contextmanager
def use_export_safe_concat(model):
    original = model.concat_and_pad_script
    validate_export_safe_concat(original)
    model.concat_and_pad_script = export_safe_concat_and_pad
    try:
        yield
    finally:
        model.concat_and_pad_script = original


def parity_inputs() -> dict[str, Any]:
    import numpy as np

    def fixed_values(shape: tuple[int, ...], scale: float, offset: int):
        count = int(np.prod(shape))
        values = (np.arange(count, dtype=np.uint32) + offset) % 257
        return ((values.astype(np.float32) - 128.0) * scale).reshape(shape)

    batch_size = 2
    feature_frames = (CHUNK_LEN + RIGHT_CONTEXT) * SUBSAMPLING_FACTOR
    return {
        "chunk": fixed_values((batch_size, feature_frames, MEL_BINS), 1e-3, 0),
        "chunk_lengths": np.asarray(
            [feature_frames, feature_frames - 24], dtype=np.int64
        ),
        "spkcache": fixed_values(
            (batch_size, SPEAKER_CACHE_LEN, EMBEDDING_DIM), 5e-4, 31
        ),
        "spkcache_lengths": np.asarray([5, 0], dtype=np.int64),
        "fifo": fixed_values((batch_size, FIFO_LEN, EMBEDDING_DIM), 5e-4, 73),
        "fifo_lengths": np.asarray([7, 3], dtype=np.int64),
    }


def runtime_parity_inputs(feature_frames: int, offset: int) -> dict[str, Any]:
    import numpy as np

    count = feature_frames * MEL_BINS
    values = (np.arange(count, dtype=np.uint32) + offset) % 257
    chunk = ((values.astype(np.float32) - 128.0) * 1e-3).reshape(
        1, feature_frames, MEL_BINS
    )
    return {
        "chunk": chunk,
        "chunk_lengths": np.asarray([feature_frames], dtype=np.int64),
        "spkcache": np.zeros((1, 1, EMBEDDING_DIM), dtype=np.float32),
        "spkcache_lengths": np.asarray([0], dtype=np.int64),
        "fifo": np.zeros((1, 1, EMBEDDING_DIM), dtype=np.float32),
        "fifo_lengths": np.asarray([0], dtype=np.int64),
    }


def export_raw(model, raw_path: Path) -> None:
    import torch

    patch_torch_onnx_export()
    dynamic_axes = {
        "chunk": {0: "batch", 1: "chunk_frames"},
        "chunk_lengths": {0: "batch"},
        "spkcache": {0: "batch", 1: "spkcache_frames"},
        "spkcache_lengths": {0: "batch"},
        "fifo": {0: "batch", 1: "fifo_frames"},
        "fifo_lengths": {0: "batch"},
        "spkcache_fifo_chunk_preds": {0: "batch", 1: "prediction_frames"},
        "chunk_pre_encode_embs": {0: "batch", 1: "chunk_embedding_frames"},
        "chunk_pre_encode_lengths": {0: "batch"},
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = parity_inputs()
    input_example = tuple(torch.from_numpy(inputs[name]) for name in INPUT_CONTRACT)
    with use_export_safe_concat(model):
        model.export(
            output=str(raw_path),
            input_example=input_example,
            dynamic_axes=dynamic_axes,
            check_trace=False,
        )
    if not raw_path.exists():
        raise RuntimeError(f"NeMo reported success but did not create {raw_path}")


def set_metadata(graph, metadata: dict[str, str]):
    import onnx

    existing = {entry.key: entry.value for entry in graph.metadata_props}
    existing.update(metadata)
    onnx.helper.set_model_props(graph, existing)
    return graph


def save_external_model(
    graph,
    staged_path: Path,
    data_filename: str,
    metadata: dict[str, str],
) -> None:
    import onnx

    set_metadata(graph, metadata)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = staged_path.with_name(data_filename)
    staged_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    onnx.save_model(
        graph,
        str(staged_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
    )
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export did not create external data file {data_path}")


def convert_to_fp16(graph):
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(
        graph,
        keep_io_types=True,
        disable_shape_infer=False,
    )


def tensor_element_type(value_info) -> str:
    import onnx

    element_type = value_info.type.tensor_type.elem_type
    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.INT64: "int64",
    }
    return names.get(element_type, f"onnx_type_{element_type}")


def static_dimension(value_info, axis: int) -> int | None:
    dimension = value_info.type.tensor_type.shape.dim[axis]
    return dimension.dim_value if dimension.HasField("dim_value") else None


def validate_tensors(kind: str, values, expected: dict[str, tuple[str, int]]) -> None:
    by_name = {value.name: value for value in values}
    if list(by_name) != list(expected):
        raise RuntimeError(
            f"Unexpected ONNX {kind} names/order: expected {list(expected)}, "
            f"found {list(by_name)}"
        )
    for name, (expected_type, expected_rank) in expected.items():
        value = by_name[name]
        actual_type = tensor_element_type(value)
        actual_rank = len(value.type.tensor_type.shape.dim)
        if actual_type != expected_type or actual_rank != expected_rank:
            raise RuntimeError(
                f"Unexpected {kind} {name}: expected {expected_type} rank "
                f"{expected_rank}, found {actual_type} rank {actual_rank}"
            )
        if static_dimension(value, 0) is not None:
            raise RuntimeError(f"ONNX {kind} {name} has a fixed batch dimension")


def external_data_location(initializer) -> str | None:
    return next(
        (entry.value for entry in initializer.external_data if entry.key == "location"),
        None,
    )


def validate_model(
    path: Path,
    data_filename: str,
    expected_metadata: dict[str, str],
) -> None:
    import onnx

    onnx.checker.check_model(str(path), full_check=True)
    graph = onnx.load_model(str(path), load_external_data=False)
    validate_tensors("input", graph.graph.input, INPUT_CONTRACT)
    validate_tensors("output", graph.graph.output, OUTPUT_CONTRACT)

    inputs = {value.name: value for value in graph.graph.input}
    outputs = {value.name: value for value in graph.graph.output}
    dimensions = [
        (inputs["chunk"], 2, MEL_BINS),
        (inputs["spkcache"], 2, EMBEDDING_DIM),
        (inputs["fifo"], 2, EMBEDDING_DIM),
        (outputs["spkcache_fifo_chunk_preds"], 2, MAX_SPEAKERS),
        (outputs["chunk_pre_encode_embs"], 2, EMBEDDING_DIM),
    ]
    for value, axis, expected in dimensions:
        actual = static_dimension(value, axis)
        if actual is not None and actual != expected:
            raise RuntimeError(
                f"Unexpected dimension for {value.name} axis {axis}: "
                f"expected {expected}, found {actual}"
            )

    external_initializers = [
        initializer
        for initializer in graph.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
    ]
    if not external_initializers:
        raise RuntimeError(f"ONNX model {path} does not use external tensor data")
    locations = {external_data_location(value) for value in external_initializers}
    if locations != {data_filename}:
        raise RuntimeError(
            f"ONNX model {path} references unexpected external data {locations}"
        )
    data_path = path.with_name(data_filename)
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise RuntimeError(f"Missing external tensor data {data_path}")

    actual_metadata = {entry.key: entry.value for entry in graph.metadata_props}
    missing = {
        key: value
        for key, value in expected_metadata.items()
        if actual_metadata.get(key) != value
    }
    if missing:
        raise RuntimeError(f"ONNX custom metadata did not round-trip: {missing}")


def validate_fp16_internals(fp32_path: Path, fp16_path: Path) -> None:
    import onnx

    fp32_graph = onnx.load_model(str(fp32_path), load_external_data=False)
    fp16_graph = onnx.load_model(str(fp16_path), load_external_data=False)
    fp32_float_names = {
        value.name
        for value in fp32_graph.graph.initializer
        if value.data_type == onnx.TensorProto.FLOAT
    }
    fp16_float_names = {
        value.name
        for value in fp16_graph.graph.initializer
        if value.data_type == onnx.TensorProto.FLOAT16
    }
    converted = fp32_float_names & fp16_float_names
    if not converted:
        raise RuntimeError("FP16 model contains no converted floating-point weights")

    quantized_types = {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
    fp32_quantized = {
        value.name
        for value in fp32_graph.graph.initializer
        if value.data_type in quantized_types
    }
    fp16_quantized = {
        value.name
        for value in fp16_graph.graph.initializer
        if value.data_type in quantized_types
    }
    if not fp16_quantized.issubset(fp32_quantized):
        raise RuntimeError("FP16 conversion unexpectedly introduced integer weights")

    quantization_ops = {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}
    fp32_quantization_nodes = sum(
        node.op_type in quantization_ops for node in fp32_graph.graph.node
    )
    fp16_quantization_nodes = sum(
        node.op_type in quantization_ops for node in fp16_graph.graph.node
    )
    if fp16_quantization_nodes > fp32_quantization_nodes:
        raise RuntimeError("FP16 conversion unexpectedly introduced quantization operators")


def validate_cpu_parity(model, path: Path) -> None:
    import numpy as np
    import onnxruntime
    import torch

    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    cases = [
        ("stateful-104", parity_inputs()),
        ("empty-partial-81", runtime_parity_inputs(81, 17)),
        ("empty-left-context-112", runtime_parity_inputs(112, 29)),
    ]
    for case, inputs in cases:
        torch_inputs = tuple(torch.from_numpy(inputs[name]) for name in INPUT_CONTRACT)
        with torch.inference_mode():
            expected_tensors = model.forward_for_export(*torch_inputs)
        expected = [tensor.detach().cpu().numpy() for tensor in expected_tensors]
        actual = session.run(list(OUTPUT_CONTRACT), inputs)

        for name, expected_value, actual_value in zip(
            OUTPUT_CONTRACT, expected, actual, strict=True
        ):
            if expected_value.shape != actual_value.shape:
                raise RuntimeError(
                    f"CPU parity failed for {case}/{name}: NeMo shape "
                    f"{expected_value.shape}, ONNX Runtime shape {actual_value.shape}"
                )
            try:
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    atol=PARITY_ATOL,
                    rtol=PARITY_RTOL,
                )
            except AssertionError as error:
                raise RuntimeError(
                    f"CPU parity failed for {case}/{name} "
                    f"(atol={PARITY_ATOL}, rtol={PARITY_RTOL}): {error}"
                ) from error


def write_model_source(
    dest: Path,
    resolved_revision: str,
    export_environment: dict[str, str],
) -> None:
    environment = ", ".join(
        f"{name}={version}" for name, version in export_environment.items()
    )
    content = (
        f"Source model: {MODEL_ID}\n"
        f"Source revision: {resolved_revision}\n"
        f"Upstream URL: https://huggingface.co/{MODEL_ID}\n"
        f"Pinned revision: https://huggingface.co/{MODEL_ID}/tree/{resolved_revision}\n"
        f"Upstream license: {UPSTREAM_LICENSE}\n"
        f"Exported with NeMo: {NEMO_VERSION}\n"
        f"Conversion environment: {environment}\n"
        "FP32 artifact: internal FP32 weights and FP32 external I/O\n"
        "FP16 artifact: internal FP16 floating-point weights/operators with FP32 external I/O\n"
        "FP16 conversion is not integer quantization.\n"
        f"Streaming preset: low_latency ({LOW_LATENCY_MS} ms input context)\n"
        "Runtime format: ONNX neural streaming Sortformer contract\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8", newline="\n")


def install_staged_output(
    staged_dir: Path,
    output_dir: Path,
    keep_staged: bool = False,
) -> list[Path]:
    missing = [
        name
        for name in OUTPUT_FILENAMES
        if not (staged_dir / name).is_file() or (staged_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Staged export is incomplete; missing {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for name in OUTPUT_FILENAMES:
        source = staged_dir / name
        destination = output_dir / name
        if keep_staged:
            temporary = output_dir / f".{name}.installing"
            temporary.unlink(missing_ok=True)
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            os.replace(source, destination)
        installed.append(destination)
    return installed


def run() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    work_dir = args.work_dir.resolve()
    raw_dir = work_dir / "raw"
    staged_dir = work_dir / "staged"
    raw_path = raw_dir / "sortformer.raw.onnx"
    fp32_path = staged_dir / FP32_MODEL_FILENAME
    fp16_path = staged_dir / FP16_MODEL_FILENAME

    cache_directories = configure_local_caches(cache_dir)
    validate_output_paths(output_dir, work_dir, cache_directories)
    export_environment = require_export_environment()

    print(f"Downloading pinned model: {MODEL_ID}@{MODEL_REVISION}")
    archive, resolved_revision = download_source_model(cache_dir)
    print(f"Loading NeMo archive: {archive}")
    model = load_model(archive)
    base_metadata = model_metadata(model, resolved_revision, export_environment)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    print(f"Exporting raw ONNX: {raw_path}")
    export_raw(model, raw_path)

    import onnx

    print("Writing FP32 model with external tensor data")
    fp32_metadata = {**base_metadata, **precision_metadata("fp32")}
    save_external_model(
        onnx.load_model(str(raw_path), load_external_data=True),
        fp32_path,
        FP32_DATA_FILENAME,
        fp32_metadata,
    )

    print("Converting internal floating-point weights and operators to FP16")
    fp16_metadata = {**base_metadata, **precision_metadata("fp16")}
    fp16_graph = convert_to_fp16(
        onnx.load_model(str(raw_path), load_external_data=True)
    )
    save_external_model(
        fp16_graph,
        fp16_path,
        FP16_DATA_FILENAME,
        fp16_metadata,
    )

    write_model_source(
        staged_dir / SOURCE_FILENAME,
        resolved_revision,
        export_environment,
    )
    validate_model(fp32_path, FP32_DATA_FILENAME, fp32_metadata)
    validate_model(fp16_path, FP16_DATA_FILENAME, fp16_metadata)
    validate_fp16_internals(fp32_path, fp16_path)
    print(
        "Checking FP32 ONNX Runtime CPU parity "
        f"(atol={PARITY_ATOL}, rtol={PARITY_RTOL})"
    )
    validate_cpu_parity(model, fp32_path)

    installed = install_staged_output(
        staged_dir,
        output_dir,
        keep_staged=args.keep_work_dir,
    )
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)

    print("Done")
    for path in installed:
        print(f"  {path}")


if __name__ == "__main__":
    run()
