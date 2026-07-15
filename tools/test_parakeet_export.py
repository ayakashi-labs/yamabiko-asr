from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import parakeet_export
from parakeet_export import ExportDefaults, export_raw, tokenizer_pieces, write_vocab


class PieceTokenizer:
    def __init__(self, pieces):
        self._pieces = pieces

    def get_piece_size(self):
        return len(self._pieces)

    def id_to_piece(self, index):
        return self._pieces[index]


def model_with_pieces(pieces, vocab_size):
    return SimpleNamespace(
        tokenizer=PieceTokenizer(pieces),
        decoder=SimpleNamespace(vocab_size=vocab_size),
    )


class ExportRawTests(unittest.TestCase):
    def test_passes_check_trace_when_export_supports_it(self):
        class Model:
            def __init__(self):
                self.check_trace = None

            def export(self, *, output, check_trace):
                self.check_trace = check_trace
                Path(output).touch()

        with TemporaryDirectory() as directory:
            model = Model()
            export_raw(model, Path(directory))

        self.assertFalse(model.check_trace)

    def test_omits_check_trace_when_export_does_not_support_it(self):
        class Model:
            def __init__(self):
                self.output = None

            def export(self, *, output):
                self.output = output
                Path(output).touch()

        with TemporaryDirectory() as directory:
            model = Model()
            export_raw(model, Path(directory))

        self.assertIsNotNone(model.output)

    def test_does_not_retry_internal_type_error(self):
        class Model:
            def __init__(self):
                self.calls = 0

            def export(self, *, output, check_trace):
                self.calls += 1
                raise TypeError("internal export failure")

        with TemporaryDirectory() as directory:
            model = Model()
            with self.assertRaisesRegex(TypeError, "internal export failure"):
                export_raw(model, Path(directory))

        self.assertEqual(model.calls, 1)


class VocabularyTests(unittest.TestCase):
    def test_appends_final_blank_for_tokenizer_only_vocabulary(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "vocab.txt"
            write_vocab(model_with_pieces(["a", "b"], 2), destination)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "a 0\nb 1\n<blank> 2\n",
            )

    def test_accepts_existing_final_blank(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "vocab.txt"
            write_vocab(
                model_with_pieces(["a", "<blank>"], 1),
                destination,
            )
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "a 0\n<blank> 1\n",
            )

    def test_rejects_incompatible_decoder_vocabulary_size(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "vocab.txt"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                write_vocab(model_with_pieces(["a", "b"], 9), destination)
            self.assertFalse(destination.exists())

    def test_rejects_nonfinal_blank(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "vocab.txt"
            with self.assertRaisesRegex(RuntimeError, "final ID"):
                write_vocab(
                    model_with_pieces(["<blank>", "a"], 1),
                    destination,
                )

    def test_rejects_sparse_dictionary_ids(self):
        model = SimpleNamespace(tokenizer=SimpleNamespace(vocab={"a": 0, "b": 2}))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            tokenizer_pieces(model)


class WorkDirectoryTests(unittest.TestCase):
    def test_cleanup_removes_only_the_script_owned_child(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            work_root = root / "work"
            output_dir = work_root / "output"
            work_root.mkdir()
            sentinel = work_root / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            defaults = ExportDefaults(
                model="example/model",
                output_dir=output_dir,
                work_dir=work_root,
                description="test",
            )
            args = SimpleNamespace(
                model="example/model",
                output_dir=output_dir,
                work_dir=work_root,
                keep_work_dir=False,
            )

            def fake_export_raw(_model, raw_dir):
                raw_dir.mkdir(parents=True)
                (raw_dir / "encoder.onnx").touch()
                (raw_dir / "decoder_joint.onnx").touch()

            def fake_save(_source, destination, *_args):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.touch()

            with (
                mock.patch.object(parakeet_export, "parse_args", return_value=args),
                mock.patch.object(parakeet_export, "configure_local_caches"),
                mock.patch.object(parakeet_export, "patch_torch_onnx_export"),
                mock.patch.object(parakeet_export, "load_model", return_value=object()),
                mock.patch.object(parakeet_export, "export_raw", side_effect=fake_export_raw),
                mock.patch.object(
                    parakeet_export,
                    "save_external_data_onnx",
                    side_effect=fake_save,
                ),
                mock.patch.object(
                    parakeet_export,
                    "save_single_file_onnx",
                    side_effect=fake_save,
                ),
                mock.patch.object(parakeet_export, "write_vocab", side_effect=fake_save),
                mock.patch.object(
                    parakeet_export,
                    "write_model_source",
                    side_effect=lambda *_args: None,
                ),
            ):
                parakeet_export.run_export(defaults)

            self.assertTrue(sentinel.exists())
            self.assertTrue((output_dir / "encoder.onnx").exists())
            self.assertFalse(
                any(
                    path.name.startswith("yamabiko-parakeet-export-")
                    for path in work_root.iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
