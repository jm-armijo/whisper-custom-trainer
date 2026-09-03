"""Runs the recorder as a real terminal application.

Every scenario here failed to be caught by the stub-screen tests at some point:
arrow keys silently did nothing because keypad() did not translate the escape
sequence, and the stub could not prove curses painted anything at all. These
drive the real binary through a pty and read what the terminal received.
"""

import json

import pytest

from tests.e2e.conftest import KEY_DOWN, SPACE

pytestmark = pytest.mark.e2e

TWO_LINES = (
    "El rapido zorro marron salta sobre el perro perezoso cada manana temprano. "
    "La programacion en Python es divertida y muy util para muchas tareas hoy."
)
ACCENTED = "¿Cómo estás amigo? Añejo pequeño corazón español para probar la pantalla."


class TestStartupRendering:
    """The stub proves the UI calls curses; only this proves curses paints."""

    def test_the_script_text_reaches_the_terminal(self, recorder):
        app = recorder(TWO_LINES)
        assert "rapido zorro" in app.screen

    def test_accented_text_is_not_mangled(self, recorder):
        """A locale not set before initscr renders these as replacement chars."""
        app = recorder(ACCENTED)
        assert "¿Cómo estás" in app.screen


class TestArrowKeyNavigation:
    """The regression that the stub tests could not see: keypad() does not
    always fold an arrow into KEY_DOWN, so raw ESC [ B must be decoded."""

    def test_arrow_down_then_record_saves_the_second_line(
        self, recorder, clips
    ):
        app = recorder(TWO_LINES)
        app.press(KEY_DOWN)
        app.press(SPACE, settle=1.5)
        app.press(SPACE, settle=1.5)
        app.press(b"q")
        app.wait_for_exit()

        assert clips() == ["es_00001.wav"]

class TestRecordingLifecycle:
    def test_a_take_is_written_to_disk(self, recorder, clips):
        app = recorder(TWO_LINES)
        app.press(SPACE, settle=1.5)
        app.press(SPACE, settle=1.5)
        app.press(b"q")
        app.wait_for_exit()

        assert clips() == ["es_00000.wav"]

    def test_the_dataset_row_carries_the_chunk_text(self, recorder, dataset_rows):
        app = recorder(TWO_LINES)
        app.press(SPACE, settle=1.5)
        app.press(SPACE, settle=1.5)
        app.press(b"q")
        app.wait_for_exit()

        assert "rapido zorro" in dataset_rows()[0]["text"]

class TestBlinkingIndicator:
    """The dot is blinked by redrawing; A_BLINK would render as a solid dot."""

    @pytest.fixture
    def fast_theme(self, tmp_path):
        path = tmp_path / "fast.json"
        path.write_text(json.dumps({"blink_ms": 300}), encoding="utf8")
        return path

    def test_the_dot_alternates_more_than_once(self, recorder, fast_theme):
        """A single toggle could be a coincidence of timing; several is a blink."""
        app = recorder(TWO_LINES, theme=fast_theme)
        app.press(SPACE, settle=3.0)

        assert app.screen.count("●") >= 2

class TestFinishedLineStaysSelected:
    """A saved line must stop reading as 'read this next' while still selected.

    The cursor does not move after a take, so the only cue available is the
    line's own styling; this is the tier that can see what colour it actually
    became.
    """

    def test_it_is_not_styled_like_the_unrecorded_line_below(self, recorder):
        app = recorder(TWO_LINES)
        app.press(SPACE, settle=1.5)
        app.press(SPACE, settle=1.5)

        assert app.styles_before("rapido zorro")[-1] != (
            app.styles_before("programacion en Python")[-1]
        )


class TestQuitting:
    def test_a_take_abandoned_by_quitting_is_not_committed(
        self, recorder, dataset_rows
    ):
        app = recorder(TWO_LINES)
        app.press(SPACE, settle=1.5)
        app.press(b"q", settle=0.2)
        app.wait_for_exit()

        assert dataset_rows() == []


class TestResumeAcrossRuns:
    def test_a_second_run_opens_on_the_next_unrecorded_line(
        self, recorder, clips
    ):
        first = recorder(TWO_LINES)
        first.press(SPACE, settle=1.5)
        first.press(SPACE, settle=1.5)
        first.press(b"q")
        first.wait_for_exit()

        second = recorder(TWO_LINES)
        second.press(SPACE, settle=1.5)
        second.press(SPACE, settle=1.5)
        second.press(b"q")
        second.wait_for_exit()

        assert clips() == ["es_00000.wav", "es_00001.wav"]

class TestReRecordConfirmation:
    def test_confirming_does_not_duplicate_the_row(self, recorder, dataset_rows):
        app = recorder(TWO_LINES)
        app.press(SPACE, settle=1.5)
        app.press(SPACE, settle=1.5)

        app.press(SPACE, settle=0.8)
        app.press(b"y", settle=0.8)
        app.press(SPACE, settle=1.5)
        app.press(b"q")
        app.wait_for_exit()

        assert len(dataset_rows()) == 1


class TestErrorPaths:
    """A bad invocation must fail with a message, not a curses traceback."""

    def test_an_unusable_script_shows_no_traceback(self, recorder):
        app = recorder("   \n\n  \n")
        app.wait_for_exit()

        assert "Traceback" not in app.screen

class TestNarrowTerminal:
    def test_a_narrow_window_does_not_crash(self, recorder):
        app = recorder(TWO_LINES, columns=24, lines=10)
        app.press(KEY_DOWN)
        app.press(b"q", settle=0.3)

        assert app.wait_for_exit()
