"""Drives record_data end to end with a stubbed microphone and real file writes."""

import csv

import numpy as np
import pytest

import record_data
import whisper_pipeline as wp

pytestmark = pytest.mark.integration


@pytest.fixture
def fake_microphone(monkeypatch):
    """Replace the mic with silence; everything else stays real."""
    monkeypatch.setattr(
        record_data, "record_clip",
        lambda: np.zeros(int(wp.SAMPLE_RATE * 1.5), dtype="float32"),
    )


@pytest.fixture
def keystrokes(monkeypatch):
    """Feed a scripted sequence of console inputs."""
    def script(responses):
        pending = iter(responses)
        monkeypatch.setattr("builtins.input", lambda *_: next(pending, "q"))
    return script


@pytest.fixture
def session(tmp_path):
    from argparse import Namespace

    return Namespace(
        out_dir=tmp_path / "data",
        csv=tmp_path / "dataset.csv",
        lang="es",
    )


def read_rows(csv_path):
    with csv_path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


class TestRecordSession:
    def test_writes_one_wav_and_one_row_per_chunk(
        self, session, fake_microphone, keystrokes
    ):
        session.out_dir.mkdir(parents=True)
        keystrokes(["", "", "", ""])

        record_data.record_session(["uno dos tres", "cuatro cinco seis"], 0, session)

        assert len(list(session.out_dir.glob("*.wav"))) == 2
        assert len(read_rows(session.csv)) == 2

    def test_saved_audio_is_16khz_mono(self, session, fake_microphone, keystrokes):
        import soundfile as sf

        session.out_dir.mkdir(parents=True)
        keystrokes(["", ""])

        record_data.record_session(["uno dos tres"], 0, session)

        info = sf.info(str(next(session.out_dir.glob("*.wav"))))
        assert (info.samplerate, info.channels) == (wp.SAMPLE_RATE, 1)

    def test_skip_leaves_no_recording(self, session, fake_microphone, keystrokes):
        session.out_dir.mkdir(parents=True)
        keystrokes(["s"])

        record_data.record_session(["uno dos tres"], 0, session)

        assert not session.csv.exists()

    def test_quit_stops_before_recording(self, session, fake_microphone, keystrokes):
        session.out_dir.mkdir(parents=True)
        keystrokes(["q"])

        record_data.record_session(["uno", "dos"], 0, session)

        assert not session.csv.exists()

    def test_redo_replaces_the_take(self, session, fake_microphone, keystrokes):
        session.out_dir.mkdir(parents=True)
        # start, redo, start, keep
        keystrokes(["", "r", "", ""])

        record_data.record_session(["uno dos tres"], 0, session)

        assert len(read_rows(session.csv)) == 1

    def test_filenames_encode_language_and_index(
        self, session, fake_microphone, keystrokes
    ):
        session.out_dir.mkdir(parents=True)
        keystrokes(["", ""])

        record_data.record_session(["uno dos tres"], 0, session)

        assert next(session.out_dir.glob("*.wav")).name == "es_00000.wav"

    def test_resume_continues_at_the_requested_index(
        self, session, fake_microphone, keystrokes
    ):
        session.out_dir.mkdir(parents=True)
        keystrokes(["", ""])

        record_data.record_session(["uno", "dos", "tres"], 2, session)

        assert next(session.out_dir.glob("*.wav")).name == "es_00002.wav"

    def test_recorded_rows_are_countable_for_the_next_session(
        self, session, fake_microphone, keystrokes
    ):
        """A resumed run must see what the previous run wrote."""
        session.out_dir.mkdir(parents=True)
        keystrokes(["", "", "", ""])

        record_data.record_session(["uno dos", "tres cuatro"], 0, session)

        assert wp.count_recorded_chunks(session.csv, "es") == 2
