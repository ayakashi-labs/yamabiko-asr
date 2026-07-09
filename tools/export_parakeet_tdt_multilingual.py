"""Export the multilingual NVIDIA Parakeet TDT v3 NeMo model to ONNX."""

from pathlib import Path

from parakeet_export import ExportDefaults, run_export


DEFAULTS = ExportDefaults(
    model="nvidia/parakeet-tdt-0.6b-v3",
    output_dir=Path("models") / "parakeet-tdt-0.6b-v3-onnx",
    work_dir=Path("models") / ".export-work-parakeet-tdt-multilingual",
    description="Export nvidia/parakeet-tdt-0.6b-v3 to ONNX",
)


if __name__ == "__main__":
    run_export(DEFAULTS)
