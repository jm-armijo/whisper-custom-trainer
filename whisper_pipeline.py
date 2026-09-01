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


def dataset_audio_path(audio_path):
    """How a clip is named in dataset.csv: its filename, nothing more.

    An absolute path pins the dataset to the machine that recorded it. The
    container writes /data/audio/es_00000.wav, and the documented workflow
    rsyncs the clips and the CSV to the laptop, where that directory does not
    exist: train.py could not load a single row, and record_data.py's startup
    prune deleted every one of them as a clip gone missing.

    The filename is enough because every clip lives directly in one audio
    directory - recorder_state.clip_path is the only thing that names one - and
    that directory is already a configured value on both front ends.
    """
    return Path(audio_path).name


def resolve_audio_path(audio_path, audio_dir):
    """The clip a dataset row refers to, on this machine.

    An absolute path is honoured as written so datasets recorded before
    filenames were stored keep loading; anything else resolves against the
    audio directory this run was told to use.
    """
    stored = Path(audio_path)
    return stored if stored.is_absolute() else Path(audio_dir) / stored


# A break in range of the maximum is preferred over cutting mid-clause, but one
# in the first few words would leave a stub line, so only the tail is searched.
NATURAL_BREAKS = ",;:—–"
MIN_WORDS_BEFORE_BREAK = 8


def split_sentences(text):
    """Split prose into sentences, honouring Spanish inverted punctuation.

    A blank line ends a sentence even without terminal punctuation: collapsing
    every whitespace run merged a heading ending in ':' into the paragraph
    below it, which was then cut mid-clause to fit the maximum.
    """
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        # Split after .!?… only when followed by whitespace, so decimals stay intact.
        parts = re.split(r"(?<=[.!?…])\s+(?=[¿¡\"'(\[]?[^\s])", normalized)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


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
    """Break an over-long sentence, preferring a natural pause to a blind cut.

    Reading aloud from a line that ends mid-clause is awkward, so a comma or
    similar within the allowed span wins over the maximum-length boundary.
    """
    pieces = []
    remaining = list(words)
    while len(remaining) > MAX_WORDS_PER_CHUNK:
        cut = _break_point(remaining)
        pieces.append(" ".join(remaining[:cut]))
        remaining = remaining[cut:]
    if remaining:
        pieces.append(" ".join(remaining))
    return pieces


def _break_point(words):
    """Index to cut at: the last natural pause in range, else the maximum."""
    for index in range(MAX_WORDS_PER_CHUNK, MIN_WORDS_BEFORE_BREAK - 1, -1):
        if words[index - 1].endswith(tuple(NATURAL_BREAKS)):
            return index
    return MAX_WORDS_PER_CHUNK


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
