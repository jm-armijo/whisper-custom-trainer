"""The converters fail obscurely on missing files, so preconditions are checked here."""

import pytest

import whisper_pipeline as wp


def populate(directory, filenames):
    directory.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (directory / name).write_text("{}")
    return directory


class TestVerifyConverterInputs:
    def test_accepts_a_directory_with_every_required_file(self, tmp_path):
        model_dir = populate(tmp_path / "model", wp.LEGACY_TOKENIZER_FILES)
        wp.verify_converter_inputs(model_dir)

    def test_rejects_a_directory_missing_vocab(self, tmp_path):
        incomplete = [f for f in wp.LEGACY_TOKENIZER_FILES if f != "vocab.json"]
        model_dir = populate(tmp_path / "model", incomplete)

        with pytest.raises(wp.PipelineError, match="vocab.json"):
            wp.verify_converter_inputs(model_dir)

    def test_rejects_a_directory_missing_preprocessor_config(self, tmp_path):
        """ct2-transformers-converter requires this exact filename."""
        incomplete = [f for f in wp.LEGACY_TOKENIZER_FILES if f != "preprocessor_config.json"]
        model_dir = populate(tmp_path / "model", incomplete)

        with pytest.raises(wp.PipelineError, match="preprocessor_config.json"):
            wp.verify_converter_inputs(model_dir)

    def test_error_names_every_missing_file(self, tmp_path):
        model_dir = populate(tmp_path / "model", ["vocab.json"])

        with pytest.raises(wp.PipelineError) as error:
            wp.verify_converter_inputs(model_dir)
        assert "added_tokens.json" in str(error.value)

    def test_error_points_at_the_recovery_step(self, tmp_path):
        model_dir = populate(tmp_path / "model", [])

        with pytest.raises(wp.PipelineError, match="merge.py"):
            wp.verify_converter_inputs(model_dir)


class TestRestoreLegacyTokenizerFiles:
    def test_rejects_a_directory_that_does_not_exist(self, tmp_path):
        with pytest.raises(wp.PipelineError, match="not found"):
            wp.restore_legacy_tokenizer_files(tmp_path / "absent")

    def test_copies_every_required_file_from_the_base_snapshot(self, tmp_path, monkeypatch):
        source = populate(tmp_path / "snapshot", wp.LEGACY_TOKENIZER_FILES)
        destination = tmp_path / "model"
        destination.mkdir()

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda *a, **k: str(source)
        )
        wp.restore_legacy_tokenizer_files(destination)

        assert all((destination / name).exists() for name in wp.LEGACY_TOKENIZER_FILES)

    def test_requests_exactly_the_files_the_converters_need(self, tmp_path, monkeypatch):
        source = populate(tmp_path / "snapshot", wp.LEGACY_TOKENIZER_FILES)
        destination = tmp_path / "model"
        destination.mkdir()
        requested = {}

        def capture(repo, allow_patterns=None, **kwargs):
            requested["patterns"] = allow_patterns
            return str(source)

        monkeypatch.setattr("huggingface_hub.snapshot_download", capture)
        wp.restore_legacy_tokenizer_files(destination)

        assert set(requested["patterns"]) == set(wp.LEGACY_TOKENIZER_FILES)
