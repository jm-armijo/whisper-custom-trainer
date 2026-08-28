"""Shared boundary layer for the Whisper fine-tuning pipeline.

Every workaround for a third-party quirk lives here rather than being scattered
across the scripts, so upgrading a library means editing one function.
"""

import re
from pathlib import Path

BASE_MODEL = "openai/whisper-small"
SAMPLE_RATE = 16000

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CSV = PROJECT_ROOT / "dataset.csv"
AUDIO_DIR = PROJECT_ROOT / "data"
ADAPTER_DIR = PROJECT_ROOT / "custom-lora-adapter"
MERGED_MODEL_DIR = PROJECT_ROOT / "merged-whisper-model"
EXPORTS_DIR = PROJECT_ROOT / "exports"
WHISPER_REPO = PROJECT_ROOT / "whisper"
WHISPER_CPP_REPO = PROJECT_ROOT / "whisper.cpp"

CSV_COLUMNS = ("audio_path", "text", "language")
SUPPORTED_LANGUAGES = ("en", "es")

# Whisper truncates at 30s; 25 words keeps a comfortable margin at normal pace.
MIN_WORDS_PER_CHUNK = 10
MAX_WORDS_PER_CHUNK = 25

MIN_CLIP_SECONDS = 0.4
MAX_CLIP_SECONDS = 29.0

# whisper.cpp's converter reads these directly and transformers 5.x no longer
# writes them. The canonical HF repos still ship them from the 4.x era, and
# fine-tuning never alters the tokenizer, so copying the originals is correct
# rather than merely convenient. CTranslate2 does not need them, but they are
# restored once here so the merged model satisfies every downstream consumer.
LEGACY_TOKENIZER_FILES = (
    "vocab.json",
    "added_tokens.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
)


class PipelineError(RuntimeError):
    """Raised with an actionable message when a pipeline precondition fails."""


def split_sentences(text):
    """Split prose into sentences, honouring Spanish inverted punctuation."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    # Split after .!?… only when followed by whitespace, so decimals stay intact.
    parts = re.split(r"(?<=[.!?…])\s+(?=[¿¡\"'(\[]?[^\s])", normalized)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text):
    """Group prose into chunks of MIN..MAX words for comfortable reading aloud."""
    chunks = []
    pending_words = []

    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) > MAX_WORDS_PER_CHUNK:
            chunks.extend(_flush(pending_words))
            pending_words = []
            chunks.extend(_hard_split(words))
            continue

        # Emit what is pending before it would overflow the readable maximum.
        if len(pending_words) + len(words) > MAX_WORDS_PER_CHUNK:
            chunks.extend(_flush(pending_words))
            pending_words = []

        pending_words.extend(words)
        if len(pending_words) >= MIN_WORDS_PER_CHUNK:
            chunks.append(" ".join(pending_words))
            pending_words = []

    chunks.extend(_flush(pending_words))
    return chunks


def _flush(words):
    """Emit trailing words as a final short chunk, or nothing when empty."""
    return [" ".join(words)] if words else []


def _hard_split(words):
    """Break an over-long sentence at word boundaries into readable pieces."""
    return [
        " ".join(words[start:start + MAX_WORDS_PER_CHUNK])
        for start in range(0, len(words), MAX_WORDS_PER_CHUNK)
    ]


def count_recorded_chunks(csv_path, language):
    """Count rows already recorded for a language."""
    import csv

    path = Path(csv_path)
    if not path.exists():
        return 0

    with path.open(newline="", encoding="utf8") as handle:
        return sum(1 for row in csv.DictReader(handle) if row["language"] == language)


def load_audio(path):
    """Decode any audio file to a 16 kHz mono float array.

    datasets 5.x requires torchcodec to decode an Audio column and returns a
    decoder object rather than an array, so the pipeline keeps plain paths and
    decodes here instead.
    """
    import librosa

    samples, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return samples


def build_processor():
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(BASE_MODEL)


def resolve_device():
    """Prefer Apple's Metal backend, falling back to CPU when unavailable."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def restore_legacy_tokenizer_files(model_dir):
    """Add the tokenizer files the downstream converters require.

    See LEGACY_TOKENIZER_FILES for why these must be copied from the base repo.
    """
    import shutil

    from huggingface_hub import snapshot_download

    destination = Path(model_dir)
    if not destination.is_dir():
        raise PipelineError(f"Model directory not found: {destination}")

    source = Path(
        snapshot_download(BASE_MODEL, allow_patterns=list(LEGACY_TOKENIZER_FILES))
    )
    for filename in LEGACY_TOKENIZER_FILES:
        # allow_patterns filters silently, so a file the base repo no longer ships
        # would otherwise surface as a bare FileNotFoundError from shutil.
        if not (source / filename).exists():
            raise PipelineError(
                f"{BASE_MODEL} no longer ships {filename}; "
                "the ggml converter cannot run without it."
            )
        shutil.copy(source / filename, destination / filename)

    verify_converter_inputs(destination)


def verify_converter_inputs(model_dir):
    """Fail early and clearly instead of deep inside a vendored converter."""
    destination = Path(model_dir)
    missing = [name for name in LEGACY_TOKENIZER_FILES if not (destination / name).exists()]
    if missing:
        raise PipelineError(
            f"{destination} is missing {', '.join(missing)}. "
            "Run merge.py to rebuild the merged model."
        )
