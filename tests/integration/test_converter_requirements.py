"""Pins which converter actually needs the legacy tokenizer files.

Verified behaviour: whisper.cpp reads vocab.json directly and fails without it,
while CTranslate2 builds its own vocabulary from tokenizer.json.
"""

import subprocess
import sys

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bare_model(tmp_path_factory):
    """A model saved by transformers 5.x, without the 4.x-era tokenizer files."""
    from transformers import WhisperForConditionalGeneration

    destination = tmp_path_factory.mktemp("bare")
    WhisperForConditionalGeneration.from_pretrained(
        wp.BASE_MODEL, dtype="float32"
    ).save_pretrained(str(destination))
    wp.build_processor().save_pretrained(str(destination))
    return destination


class TestTransformers5Output:
    def test_does_not_write_the_files_ggml_requires(self, bare_model):
        """The regression that makes restore_legacy_tokenizer_files necessary."""
        assert not (bare_model / "vocab.json").exists()

    def test_writes_processor_config_not_preprocessor_config(self, bare_model):
        assert (bare_model / "processor_config.json").exists()
        assert not (bare_model / "preprocessor_config.json").exists()


class TestGgmlRequirements:
    def test_conversion_fails_without_the_restored_files(self, bare_model, tmp_path):
        converter = wp.WHISPER_CPP_REPO / "models" / "convert-h5-to-ggml.py"
        if not converter.exists():
            pytest.skip("whisper.cpp not cloned; run setup.sh")

        result = subprocess.run(
            [sys.executable, str(converter), f"{bare_model}/",
             str(wp.WHISPER_REPO), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert "vocab.json" in result.stderr

    def test_conversion_succeeds_after_restoring_them(self, bare_model, tmp_path):
        converter = wp.WHISPER_CPP_REPO / "models" / "convert-h5-to-ggml.py"
        if not converter.exists():
            pytest.skip("whisper.cpp not cloned; run setup.sh")

        wp.restore_legacy_tokenizer_files(bare_model)
        result = subprocess.run(
            [sys.executable, str(converter), f"{bare_model}/",
             str(wp.WHISPER_REPO), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert (tmp_path / "ggml-model.bin").exists(), result.stderr


class TestCtranslate2Requirements:
    def test_conversion_succeeds_without_the_legacy_files(self, tmp_path_factory):
        """CT2 needs only what transformers 5.x already writes."""
        import export

        try:
            converter = export.converter_command()
        except wp.PipelineError:
            pytest.skip("ctranslate2 not installed")

        from transformers import WhisperForConditionalGeneration

        source = tmp_path_factory.mktemp("ct2_src")
        WhisperForConditionalGeneration.from_pretrained(
            wp.BASE_MODEL, dtype="float32"
        ).save_pretrained(str(source))
        wp.build_processor().save_pretrained(str(source))

        destination = tmp_path_factory.mktemp("ct2_out") / "model"
        result = subprocess.run(
            [converter, "--model", str(source),
             "--output_dir", str(destination), "--quantization", "float16"],
            capture_output=True, text=True,
        )
        assert (destination / "model.bin").exists(), result.stderr
