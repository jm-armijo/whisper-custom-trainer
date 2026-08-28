"""Export wiring is asserted without running the multi-GB converters."""

import pytest

import whisper_pipeline as wp
import export


@pytest.fixture
def merged_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    for name in wp.LEGACY_TOKENIZER_FILES:
        (model_dir / name).write_text("{}")

    monkeypatch.setattr(wp, "MERGED_MODEL_DIR", model_dir)
    monkeypatch.setattr(export.wp, "MERGED_MODEL_DIR", model_dir)
    monkeypatch.setattr(export.wp, "EXPORTS_DIR", tmp_path / "exports")
    return model_dir


@pytest.fixture
def captured_commands(monkeypatch):
    commands = []
    monkeypatch.setattr(export, "run", lambda command: commands.append(command))
    return commands


class TestCtranslate2Export:
    def test_invokes_the_ctranslate2_converter(self, merged_model, captured_commands):
        export.wp.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        export.export_ctranslate2()
        assert captured_commands[0][0].endswith("ct2-transformers-converter")

    def test_copies_the_tokenizer_into_the_export(self, merged_model, captured_commands):
        """CTranslate2 builds its own vocabulary but consumers still want this."""
        export.wp.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        export.export_ctranslate2()
        assert "tokenizer.json" in captured_commands[0]

    def test_only_copies_files_the_merged_model_actually_contains(
        self, merged_model, captured_commands
    ):
        """ct2-transformers-converter aborts when --copy_files names a missing file."""
        export.wp.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        export.export_ctranslate2()

        command = captured_commands[0]
        copied = command[command.index("--copy_files") + 1:]
        transformers5_outputs = {"tokenizer.json", "tokenizer_config.json",
                                 "processor_config.json", "config.json"}
        assert set(copied) <= transformers5_outputs

    def test_quantizes_to_float16(self, merged_model, captured_commands):
        export.wp.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        export.export_ctranslate2()
        command = captured_commands[0]
        assert command[command.index("--quantization") + 1] == "float16"


class TestConverterLookup:
    """The converter is a console script, so a bare name only resolves when the
    venv is on PATH. Running venv/bin/python without activating gave a raw
    FileNotFoundError from subprocess instead of an actionable message."""

    def test_prefers_the_script_beside_the_running_interpreter(
        self, tmp_path, monkeypatch
    ):
        binary = tmp_path / "ct2-transformers-converter"
        binary.write_text("")
        monkeypatch.setattr(export.sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(export.shutil, "which", lambda _: "/usr/bin/decoy")

        assert export.converter_command() == str(binary)

    def test_falls_back_to_path_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(export.sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(export.shutil, "which", lambda _: "/usr/bin/ct2")

        assert export.converter_command() == "/usr/bin/ct2"

    def test_missing_converter_raises_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(export.sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(export.shutil, "which", lambda _: None)

        with pytest.raises(wp.PipelineError, match="setup.sh"):
            export.converter_command()


class TestGgmlExport:
    def test_fails_clearly_when_whisper_cpp_is_absent(self, merged_model, tmp_path, monkeypatch):
        monkeypatch.setattr(export.wp, "WHISPER_CPP_REPO", tmp_path / "absent")
        with pytest.raises(wp.PipelineError, match="setup.sh"):
            export.export_ggml()

    def test_renames_converter_output_to_the_custom_name(
        self, merged_model, tmp_path, monkeypatch, captured_commands
    ):
        converter = tmp_path / "whisper.cpp" / "models"
        converter.mkdir(parents=True)
        (converter / "convert-h5-to-ggml.py").write_text("")
        monkeypatch.setattr(export.wp, "WHISPER_CPP_REPO", tmp_path / "whisper.cpp")

        exports = export.wp.EXPORTS_DIR
        exports.mkdir(parents=True, exist_ok=True)
        # The converter always writes this fixed filename.
        (exports / "ggml-model.bin").write_bytes(b"stub")

        assert export.export_ggml().name == export.GGML_BINARY_NAME


class TestPreflight:
    def test_export_refuses_an_incomplete_merged_model(self, tmp_path, monkeypatch):
        empty = tmp_path / "merged"
        empty.mkdir()
        monkeypatch.setattr(export.wp, "MERGED_MODEL_DIR", empty)
        monkeypatch.setattr(export.sys, "argv", ["export.py", "--format", "ct2"])

        with pytest.raises(wp.PipelineError):
            export.main()
