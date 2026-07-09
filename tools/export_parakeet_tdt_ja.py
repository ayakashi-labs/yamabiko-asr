"""Export the Japanese NVIDIA Parakeet TDT-CTC NeMo model to ONNX."""

from pathlib import Path

from parakeet_export import ExportDefaults, run_export


DEFAULTS = ExportDefaults(
    model="nvidia/parakeet-tdt_ctc-0.6b-ja",
    output_dir=Path("models") / "parakeet-tdt_ctc-0.6b-ja-onnx",
    work_dir=Path("models") / ".export-work-parakeet-tdt-ja",
    description="Export nvidia/parakeet-tdt_ctc-0.6b-ja to ONNX",
)


if __name__ == "__main__":
    run_export(DEFAULTS)
