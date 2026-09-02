"""Shared fixtures. Tests never touch the network, a microphone, or a real model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import whisper_pipeline as wp


@pytest.fixture
def csv_path(tmp_path):
    return tmp_path / "dataset.csv"


@pytest.fixture
def write_csv(csv_path):
    """Build a dataset.csv from (text, language) rows."""
    import csv as csv_module

    def build(rows):
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(wp.CSV_COLUMNS)
            for index, (text, language) in enumerate(rows):
                writer.writerow([f"data/{language}_{index:05d}.wav", text, language])
        return csv_path

    return build


@pytest.fixture
def wav_factory(tmp_path):
    """Write a real wav file at a chosen sample rate and duration."""
    import numpy as np
    import soundfile as sf

    def build(name="clip.wav", sample_rate=16000, seconds=1.0, channels=1):
        samples = np.sin(
            2 * np.pi * 440 * np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        ).astype("float32")
        if channels == 2:
            samples = np.stack([samples, samples], axis=1)

        destination = tmp_path / name
        sf.write(str(destination), samples, sample_rate)
        return destination

    return build


def pytest_collection_modifyitems(config, items):
    """Mark each test by the tier directory it lives in."""
    for item in items:
        for tier in ("unit", "integration", "e2e"):
            if f"/tests/{tier}/" in str(item.fspath):
                item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch, for fixtures that build expensive artifacts once."""
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


LINE = "this is a deliberately long sentence with plenty of words in it number {n}."


def script_of(sentences):
    return "\n\n".join(LINE.format(n=n) for n in range(sentences))


@pytest.fixture
def paths(tmp_path):
    """The three directories the server is configured with.

    Shared: the unit tier asserts the save/delete rules against it and the
    integration tier decodes a real WebM take through the same layout.
    """
    import recorder_server as srv

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "es.txt").write_text(script_of(4), encoding="utf8")
    (scripts / "en.txt").write_text(script_of(3), encoding="utf8")

    audio = tmp_path / "data"
    audio.mkdir()
    return srv.Config(
        scripts_dir=scripts,
        csv_path=tmp_path / "dataset.csv",
        audio_dir=audio,
        static_dir=tmp_path / "static",
    )
