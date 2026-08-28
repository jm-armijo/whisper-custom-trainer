"""Drives the recorder controller with a stub screen and a silent microphone."""

import csv

import numpy as np
import pytest

import record_data
import recorder_state as rs
import recorder_theme as rt
import recorder_ui as ui
import whisper_pipeline as wp

from tests.integration.test_recorder_ui import StubScreen

pytestmark = pytest.mark.integration


@pytest.fixture
def fake_microphone(monkeypatch):
    """Replace the mic with 1.5s of silence; file writing stays real."""
    monkeypatch.setattr(
        record_data, "capture_clip",
        lambda ui_, on_tick: (
            np.zeros(int(wp.SAMPLE_RATE * 1.5), dtype="float32"), "record"
        ),
    )


@pytest.fixture
def session(tmp_path):
    from argparse import Namespace

    out_dir = tmp_path / "data"
    out_dir.mkdir()
    return Namespace(out_dir=out_dir, csv=tmp_path / "dataset.csv", lang="es")


@pytest.fixture
def drive(monkeypatch, session):
    """Run the controller against scripted keys, returning the stub screen."""
    import curses

    def run(chunks, keys, recorded=None):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        screen = StubScreen()
        screen.keys = list(keys)
        theme = rt.load_theme(rt.DEFAULT_THEME_PATH)
        record_data.run(screen, chunks, session, theme)
        return screen

    return run


def rows_of(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


KEY_DOWN, KEY_UP = 258, 259     # curses.KEY_DOWN / KEY_UP
SPACE, QUIT, YES, NO = ord(" "), ord("q"), ord("y"), ord("n")


class TestRecording:
    def test_records_the_selected_chunk(self, session, fake_microphone, drive):
        drive(["uno dos tres"], [SPACE, QUIT])

        assert (session.out_dir / "es_00000.wav").exists()

    def test_writes_one_row_per_recording(self, session, fake_microphone, drive):
        drive(["uno dos tres"], [SPACE, QUIT])

        assert len(rows_of(session.csv)) == 1

    def test_saved_audio_is_16khz_mono(self, session, fake_microphone, drive):
        import soundfile as sf

        drive(["uno dos tres"], [SPACE, QUIT])

        info = sf.info(str(session.out_dir / "es_00000.wav"))
        assert (info.samplerate, info.channels) == (wp.SAMPLE_RATE, 1)

    def test_filename_encodes_language_and_index(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos"], [KEY_DOWN, SPACE, QUIT])

        assert (session.out_dir / "es_00001.wav").exists()

    def test_quit_without_recording_leaves_no_dataset(
        self, session, fake_microphone, drive
    ):
        drive(["uno dos"], [QUIT])

        assert rows_of(session.csv) == []

    def test_text_is_stored_with_the_clip(self, session, fake_microphone, drive):
        drive(["¿Cómo estás, amigo?"], [SPACE, QUIT])

        assert rows_of(session.csv)[0]["text"] == "¿Cómo estás, amigo?"


class TestNavigation:
    def test_arrow_down_moves_the_cursor(self, session, fake_microphone, drive):
        drive(["uno", "dos", "tres"], [KEY_DOWN, KEY_DOWN, SPACE, QUIT])

        assert (session.out_dir / "es_00002.wav").exists()

    def test_arrow_up_returns_to_an_earlier_line(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos"], [KEY_DOWN, KEY_UP, SPACE, QUIT])

        assert (session.out_dir / "es_00000.wav").exists()

    def test_cursor_does_not_move_above_the_first_line(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos"], [KEY_UP, KEY_UP, SPACE, QUIT])

        assert (session.out_dir / "es_00000.wav").exists()

    def test_cursor_does_not_move_past_the_last_line(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos"], [KEY_DOWN, KEY_DOWN, KEY_DOWN, SPACE, QUIT])

        assert (session.out_dir / "es_00001.wav").exists()


class TestReRecording:
    def test_confirming_overwrites_the_existing_row(
        self, session, fake_microphone, drive
    ):
        drive(["uno dos"], [SPACE, QUIT])
        drive(["uno dos"], [SPACE, YES, QUIT])

        assert len(rows_of(session.csv)) == 1

    def test_declining_leaves_the_recording_untouched(
        self, session, fake_microphone, drive
    ):
        drive(["uno dos"], [SPACE, QUIT])
        original = (session.out_dir / "es_00000.wav").read_bytes()

        drive(["uno dos"], [SPACE, NO, QUIT])

        assert (session.out_dir / "es_00000.wav").read_bytes() == original

    def test_first_recording_needs_no_confirmation(
        self, session, fake_microphone, drive
    ):
        """An unrecorded line must record on one keypress."""
        drive(["uno dos"], [SPACE, QUIT])

        assert len(rows_of(session.csv)) == 1


class TestResume:
    def test_cursor_starts_at_the_first_unrecorded_line(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos", "tres"], [SPACE, QUIT])

        # Fresh session: cursor should land on line 2, so SPACE records index 1.
        drive(["uno", "dos", "tres"], [SPACE, QUIT])

        assert (session.out_dir / "es_00001.wav").exists()

    def test_a_deleted_clip_reopens_its_line(self, session, fake_microphone, drive):
        drive(["uno", "dos"], [SPACE, QUIT])
        (session.out_dir / "es_00000.wav").unlink()

        recorded = rs.recorded_indices(session.csv, session.out_dir, "es")

        assert recorded == set()

    def test_recorded_lines_survive_a_restart(
        self, session, fake_microphone, drive
    ):
        drive(["uno", "dos"], [SPACE, QUIT])

        recorded = rs.recorded_indices(session.csv, session.out_dir, "es")

        assert recorded == {0}


class TestClipRejection:
    def test_a_clip_too_short_is_not_saved(self, session, monkeypatch, drive):
        monkeypatch.setattr(
            record_data, "capture_clip",
            lambda ui_, on_tick: (
                np.zeros(int(wp.SAMPLE_RATE * 0.1), dtype="float32"), "record"
            ),
        )

        drive(["uno dos"], [SPACE, QUIT])

        assert rows_of(session.csv) == []


class TestQuitWhileRecording:
    def test_q_during_a_take_exits_the_session(
        self, session, fake_microphone, drive
    ):
        """q must stop the take and quit, not just stop the take."""
        screen = drive(["uno dos", "tres cuatro"], [SPACE, QUIT])

        # If q only stopped the recording, the loop would still be waiting.
        assert screen.keys == []

    def test_q_during_a_take_does_not_save_the_clip(
        self, session, monkeypatch, drive
    ):
        """Quitting mid-take abandons it rather than committing a partial read."""
        monkeypatch.setattr(
            record_data, "capture_clip",
            lambda ui_, on_tick: (
                np.zeros(int(wp.SAMPLE_RATE * 1.5), dtype="float32"), "quit"
            ),
        )
        drive(["uno dos"], [SPACE])

        assert rows_of(session.csv) == []
