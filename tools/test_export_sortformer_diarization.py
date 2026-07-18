"""Dependency-free contract tests for the Streaming Sortformer exporter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_sortformer_diarization as exporter


class ExporterContractTests(unittest.TestCase):
    def test_reproducibility_and_low_latency_contract_is_pinned(self) -> None:
        self.assertEqual(
            exporter.MODEL_ID,
            "nvidia/diar_streaming_sortformer_4spk-v2.1",
        )
        self.assertEqual(
            exporter.MODEL_REVISION,
            "a494724e2261b51d18a6ef403343b1f7025b3b6d",
        )
        self.assertEqual(exporter.NEMO_VERSION, "2.7.3")
        self.assertEqual(exporter.SAMPLE_RATE, 16_000)
        self.assertEqual(exporter.MEL_BINS, 128)
        self.assertEqual(exporter.MAX_SPEAKERS, 4)
        self.assertEqual(exporter.DIARIZATION_FRAME_MS, 80)
        self.assertEqual(exporter.CHUNK_LEN, 6)
        self.assertEqual(exporter.RIGHT_CONTEXT, 7)
        self.assertEqual(exporter.FIFO_LEN, 188)
        self.assertEqual(exporter.UPDATE_PERIOD, 144)
        self.assertEqual(exporter.SPEAKER_CACHE_LEN, 188)
        self.assertEqual(exporter.LOW_LATENCY_MS, 1_040)

    def test_precision_metadata_keeps_external_io_in_fp32(self) -> None:
        fp16 = exporter.precision_metadata("fp16")
        self.assertEqual(fp16["yamabiko.model.precision"], "fp16")
        self.assertEqual(fp16["yamabiko.precision.external_io"], "float32")
        self.assertEqual(fp16["yamabiko.precision.integer_quantization"], "false")
        with self.assertRaises(ValueError):
            exporter.precision_metadata("int8")

    def test_model_source_describes_both_floating_point_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / exporter.SOURCE_FILENAME
            exporter.write_model_source(
                source,
                exporter.MODEL_REVISION,
                {"nemo_toolkit": exporter.NEMO_VERSION, "onnx": "1.0"},
            )
            content = source.read_text(encoding="utf-8")
            self.assertIn(exporter.MODEL_ID, content)
            self.assertIn(exporter.MODEL_REVISION, content)
            self.assertIn("internal FP32 weights", content)
            self.assertIn("internal FP16 floating-point weights", content)
            self.assertIn("not integer quantization", content)

    def test_model_metadata_records_frontend_tensor_and_preset_contracts(self) -> None:
        featurizer = SimpleNamespace(
            win_length=400,
            hop_length=160,
            n_fft=512,
            normalize="NA",
            preemph=0.97,
            dither=0.0,
            pad_to=16,
            pad_value=0.0,
            frame_splicing=1,
            mag_power=2.0,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2**-24,
            exact_pad=False,
        )
        modules = SimpleNamespace(
            n_spk=4,
            fc_d_model=512,
            chunk_len=6,
            chunk_right_context=7,
            chunk_left_context=1,
            fifo_len=188,
            spkcache_update_period=144,
            spkcache_len=188,
            spkcache_sil_frames_per_spk=3,
            sil_threshold=0.2,
            pred_score_threshold=0.25,
            scores_boost_latest=0.05,
            strong_boost_rate=0.75,
            weak_boost_rate=1.5,
            min_pos_scores_rate=0.5,
            subsampling_factor=8,
        )
        model = SimpleNamespace(
            cfg=SimpleNamespace(
                preprocessor={
                    "sample_rate": 16_000,
                    "features": 128,
                    "window_size": 0.025,
                    "window_stride": 0.01,
                    "window": "hann",
                    "mel_norm": "slaney",
                    "lowfreq": 0,
                    "highfreq": 8_000,
                }
            ),
            preprocessor=SimpleNamespace(featurizer=featurizer),
            sortformer_modules=modules,
        )

        metadata = exporter.model_metadata(
            model,
            exporter.MODEL_REVISION,
            {"nemo_toolkit": exporter.NEMO_VERSION, "onnx": "1.0"},
        )

        self.assertEqual(metadata["yamabiko.model.id"], exporter.MODEL_ID)
        self.assertEqual(metadata["yamabiko.features.n_mels"], "128")
        self.assertEqual(metadata["yamabiko.streaming.preset"], "low_latency")
        self.assertEqual(metadata["yamabiko.streaming.latency_ms"], "1040")
        self.assertEqual(
            metadata["yamabiko.tensor.inputs"],
            ",".join(exporter.INPUT_CONTRACT),
        )

    def test_work_directory_must_not_overlap_managed_or_current_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cwd = root / "repo"
            output = cwd / "models" / "output"
            cache = cwd / "models" / "cache"
            work = cwd / "models" / "work"
            cwd.mkdir()

            exporter.validate_output_paths(output, cache, work, cwd)
            for unsafe in [cwd, root, output, output / "work", cache / "work"]:
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(RuntimeError):
                        exporter.validate_output_paths(output, cache, unsafe, cwd)

    def test_install_replaces_all_artifacts_on_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            for name in exporter.OUTPUT_FILENAMES:
                (output / name).write_text("old", encoding="utf-8")

            for generation in ["first", "second"]:
                staged = root / f"staged-{generation}"
                staged.mkdir()
                for name in exporter.OUTPUT_FILENAMES:
                    (staged / name).write_text(generation, encoding="utf-8")
                installed = exporter.install_staged_output(staged, output)
                self.assertEqual(
                    installed,
                    [output / name for name in exporter.OUTPUT_FILENAMES],
                )
                for name in exporter.OUTPUT_FILENAMES:
                    self.assertEqual(
                        (output / name).read_text(encoding="utf-8"), generation
                    )

    def test_incomplete_stage_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            staged = root / "staged"
            output.mkdir()
            staged.mkdir()
            for name in exporter.OUTPUT_FILENAMES:
                (output / name).write_text("old", encoding="utf-8")
            for name in exporter.OUTPUT_FILENAMES[:-1]:
                (staged / name).write_text("new", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                exporter.install_staged_output(staged, output)
            for name in exporter.OUTPUT_FILENAMES:
                self.assertEqual((output / name).read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
