"""Shared NeMo-to-ONNX export helpers for Parakeet TDT models.

The output directory is shaped for this crate's local ONNX runner:

  encoder.onnx
  encoder.onnx.data
  decoder_joint.onnx
  vocab.txt

Hugging Face and NeMo caches are kept under the local models directory by
default so conversion artifacts do not leak into the repository.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportDefaults:
    model: str
    output_dir: Path
    work_dir: Path
    description: str
    upstream_license: str = "CC-BY-4.0"


def parse_args(defaults: ExportDefaults) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=defaults.description)
    parser.add_argument(
        "--model",
        default=defaults.model,
        help="Hugging Face model id or local .nemo path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="Directory to write crate-compatible ONNX files",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=defaults.work_dir,
        help="Temporary directory for raw NeMo ONNX export artifacts",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep raw export artifacts for debugging",
    )
    return parser.parse_args()


def configure_local_caches(output_dir: Path) -> None:
    cache_root = output_dir.parent
    os.environ.setdefault("HF_HOME", str((cache_root / ".hf-cache").resolve()))
    os.environ.setdefault("NEMO_CACHE_DIR", str((cache_root / ".nemo-cache").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((cache_root / ".mpl-cache").resolve()))
    os.environ.setdefault("NUMBA_CACHE_DIR", str((cache_root / ".numba-cache").resolve()))


def patch_torch_onnx_export() -> None:
    import torch

    marker = "_legacy_onnx_patched"
    if getattr(torch.onnx.export, marker, False):
        return

    original_export = torch.onnx.export

    @functools.wraps(original_export)
    def patched_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    setattr(patched_export, marker, True)
    torch.onnx.export = patched_export


def load_model(model_name_or_path: str):
    import nemo.collections.asr as nemo_asr

    model_path = Path(model_name_or_path)
    if model_path.exists():
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(model_path),
            map_location="cpu",
        )
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name_or_path,
        )

    model.eval()
    model.freeze()
    return model.cpu()


def export_raw(model, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / "model.onnx"
    try:
        parameters = inspect.signature(model.export).parameters.values()
    except (TypeError, ValueError):
        parameters = ()

    accepts_check_trace = any(
        parameter.name == "check_trace"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_check_trace:
        model.export(output=str(output), check_trace=False)
    else:
        model.export(output=str(output))


def find_single(root: Path, *patterns: str) -> Path:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(root.glob(pattern))

    onnx_matches = sorted(path for path in matches if path.suffix == ".onnx")
    if len(onnx_matches) != 1:
        found = ", ".join(str(path.name) for path in onnx_matches) or "none"
        expected = ", ".join(patterns)
        raise RuntimeError(f"Expected one file matching {expected}, found {found}")
    return onnx_matches[0]


def save_external_data_onnx(src: Path, dest: Path, data_filename: str) -> None:
    import onnx

    model = onnx.load_model(str(src), load_external_data=True)
    if dest.exists():
        dest.unlink()
    data_path = dest.with_name(data_filename)
    if data_path.exists():
        data_path.unlink()

    onnx.save_model(
        model,
        str(dest),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
    )


def save_single_file_onnx(src: Path, dest: Path) -> None:
    import onnx

    model = onnx.load_model(str(src), load_external_data=True)
    if dest.exists():
        dest.unlink()
    onnx.save_model(model, str(dest), save_as_external_data=False)


def tokenizer_pieces(model) -> list[str]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Loaded model does not expose tokenizer")

    candidates = [
        tokenizer,
        getattr(tokenizer, "tokenizer", None),
        getattr(tokenizer, "tokenizer_model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "get_piece_size") and hasattr(candidate, "id_to_piece"):
            return [
                str(candidate.id_to_piece(index))
                for index in range(candidate.get_piece_size())
            ]
        if hasattr(candidate, "vocab"):
            vocab = getattr(candidate, "vocab")
            if isinstance(vocab, dict):
                ids = list(vocab.values())
                if (
                    any(not isinstance(index, int) or isinstance(index, bool) for index in ids)
                    or len(set(ids)) != len(ids)
                    or sorted(ids) != list(range(len(ids)))
                ):
                    raise RuntimeError(
                        "Tokenizer vocabulary IDs must be unique and contiguous from zero"
                    )
                pieces = [""] * len(vocab)
                for token, index in vocab.items():
                    pieces[index] = str(token)
                return pieces
            if isinstance(vocab, list):
                return [str(token) for token in vocab]

    raise RuntimeError("Could not extract tokenizer pieces from NeMo model")


def decoder_vocab_size(model) -> int | None:
    decoder = getattr(model, "decoder", None)
    value = getattr(decoder, "vocab_size", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def write_vocab(model, dest: Path) -> None:
    pieces = tokenizer_pieces(model)
    if not pieces:
        raise RuntimeError("Tokenizer vocabulary is empty")
    if any(not piece or "\n" in piece or "\r" in piece for piece in pieces):
        raise RuntimeError("Tokenizer vocabulary contains an empty or multiline token")

    blank_markers = {"<blank>", "<blk>"}
    blank_positions = [
        index for index, piece in enumerate(pieces) if piece in blank_markers
    ]
    if blank_positions and blank_positions != [len(pieces) - 1]:
        raise RuntimeError("Tokenizer blank token must appear exactly once as the final ID")

    # NeMo transducer decoders commonly use blank_id == tokenizer vocab size.
    # This crate expects the blank as the final vocabulary row.
    vocab_size = decoder_vocab_size(model)
    if not blank_positions:
        if vocab_size is not None and vocab_size not in {len(pieces), len(pieces) + 1}:
            raise RuntimeError(
                "Decoder vocabulary size does not match the tokenizer vocabulary"
            )
        pieces.append("<blank>")
    elif vocab_size is not None and vocab_size not in {len(pieces) - 1, len(pieces)}:
        raise RuntimeError(
            "Decoder vocabulary size does not match the tokenizer vocabulary"
        )

    with dest.open("w", encoding="utf-8", newline="\n") as handle:
        for index, piece in enumerate(pieces):
            handle.write(f"{piece} {index}\n")


def write_model_source(
    model_id: str,
    output_dir: Path,
    default_model: str,
    upstream_license: str,
) -> None:
    model_url = model_id if "/" in model_id and not Path(model_id).exists() else default_model
    content = (
        f"Source model: {model_url}\n"
        f"Upstream URL: https://huggingface.co/{model_url}\n"
        f"Upstream license: {upstream_license}\n"
        f"Export input: {model_id}\n"
    )
    (output_dir / "MODEL_SOURCE.txt").write_text(content, encoding="utf-8")


def run_export(defaults: ExportDefaults) -> None:
    args = parse_args(defaults)
    output_dir = args.output_dir.resolve()
    work_root = args.work_dir.resolve()

    configure_local_caches(output_dir)
    patch_torch_onnx_export()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    work_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="yamabiko-parakeet-export-", dir=work_root))
    raw_dir = run_dir / "raw"
    try:
        if output_dir == run_dir or run_dir in output_dir.parents:
            raise RuntimeError("Output directory must not be inside temporary export data")

        print(f"Exporting raw ONNX files: {raw_dir}")
        export_raw(model, raw_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        encoder = find_single(raw_dir, "encoder*.onnx", "*encoder*.onnx")
        decoder_joint = find_single(
            raw_dir,
            "decoder_joint*.onnx",
            "*decoder_joint*.onnx",
            "*joint*.onnx",
        )

        print("Writing local ONNX model layout")
        save_external_data_onnx(
            encoder, output_dir / "encoder.onnx", "encoder.onnx.data"
        )
        save_single_file_onnx(decoder_joint, output_dir / "decoder_joint.onnx")
        write_vocab(model, output_dir / "vocab.txt")
        write_model_source(
            args.model,
            output_dir,
            defaults.model,
            defaults.upstream_license,
        )
    finally:
        if args.keep_work_dir:
            print(f"Kept export work directory: {run_dir}")
        else:
            shutil.rmtree(run_dir)

    print("Done")
    for path in [
        output_dir / "encoder.onnx",
        output_dir / "encoder.onnx.data",
        output_dir / "decoder_joint.onnx",
        output_dir / "vocab.txt",
        output_dir / "MODEL_SOURCE.txt",
    ]:
        print(f"  {path}")
