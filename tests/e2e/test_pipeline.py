"""Full pipeline: merged model -> exported formats -> transcribed speech.

These run the real converters against a real model and are slow by nature.
Run with: pytest -m e2e
"""

import shutil
import subprocess

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.e2e

SPOKEN_ENGLISH = "The quick brown fox jumps over the lazy dog"
SPOKEN_SPANISH = "Hola, estoy probando el modelo"


def require(executable):
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} not installed")


@pytest.fixture(scope="module")
def spoken_clip(tmp_path_factory):
    """Synthesise speech with macOS `say`, the only speech available offline."""
    require("say")
    require("ffmpeg")
    directory = tmp_path_factory.mktemp("speech")

    def build(text, voice):
        raw = directory / f"{voice}.aiff"
        wav = directory / f"{voice}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(raw), text], check=True)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw),
             "-ar", str(wp.SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
        return wav

    return build


@pytest.fixture(scope="module")
def merged_model(tmp_path_factory):
    """A converter-ready model directory, built the way merge.py builds it."""
    from transformers import WhisperForConditionalGeneration

    destination = tmp_path_factory.mktemp("merged")
    WhisperForConditionalGeneration.from_pretrained(
        wp.BASE_MODEL, dtype="float32"
    ).save_pretrained(str(destination))
    wp.build_processor().save_pretrained(str(destination))
    wp.restore_legacy_tokenizer_files(destination)
    return destination


@pytest.fixture(scope="module")
def exports(tmp_path_factory, merged_model, monkeypatch_module):
    directory = tmp_path_factory.mktemp("exports")
    monkeypatch_module.setattr(wp, "MERGED_MODEL_DIR", merged_model)
    monkeypatch_module.setattr(wp, "EXPORTS_DIR", directory)
    return directory


class TestCtranslate2Export:
    def test_export_produces_a_loadable_model(self, exports, merged_model):
        require("ct2-transformers-converter")
        import export as exporter

        destination = exporter.export_ctranslate2()
        assert (destination / "model.bin").exists()

    def test_faster_whisper_transcribes_english(self, exports, spoken_clip):
        faster_whisper = pytest.importorskip("faster_whisper")

        model = faster_whisper.WhisperModel(str(exports / "ct2"), device="cpu",
                                            compute_type="int8")
        segments, _ = model.transcribe(str(spoken_clip(SPOKEN_ENGLISH, "Samantha")),
                                       language="en")
        assert "quick brown fox" in " ".join(s.text for s in segments).lower()

    def test_faster_whisper_transcribes_spanish(self, exports, spoken_clip):
        faster_whisper = pytest.importorskip("faster_whisper")

        model = faster_whisper.WhisperModel(str(exports / "ct2"), device="cpu",
                                            compute_type="int8")
        segments, _ = model.transcribe(str(spoken_clip(SPOKEN_SPANISH, "Monica")),
                                       language="es")
        assert "probando" in " ".join(s.text for s in segments).lower()


class TestGgmlExport:
    def test_export_produces_a_valid_ggml_binary(self, exports, merged_model):
        import struct

        import export as exporter

        if not (wp.WHISPER_CPP_REPO / "models" / "convert-h5-to-ggml.py").exists():
            pytest.skip("whisper.cpp not cloned; run setup.sh")

        binary = exporter.export_ggml()
        with binary.open("rb") as handle:
            magic = struct.unpack("i", handle.read(4))[0]
        assert magic == 0x67676D6C

    def test_whisper_cli_transcribes_english(self, exports, spoken_clip):
        require("whisper-cli")
        binary = exports / export_binary_name()
        if not binary.exists():
            pytest.skip("ggml export not produced")

        result = subprocess.run(
            ["whisper-cli", "-m", str(binary), "-f",
             str(spoken_clip(SPOKEN_ENGLISH, "Samantha")), "-l", "en", "-nt"],
            capture_output=True, text=True, check=True,
        )
        assert "quick brown fox" in result.stdout.lower()


def export_binary_name():
    import export as exporter

    return exporter.GGML_BINARY_NAME
