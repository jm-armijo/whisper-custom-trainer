"""Renders RecorderUI against a stub screen and inspects what it drew.

A real curses.initscr() cannot run under pytest: pytest replaces sys.stdout
while curses drives the terminal fd directly, so it either errors or leaves the
terminal corrupted for the rest of the session. The stub verifies that the UI
issues the curses calls intended; that curses paints them correctly is covered
by the manual checks in the plan.
"""

import curses

import pytest

import recorder_state as rs
import recorder_theme as rt
import recorder_ui as ui

pytestmark = pytest.mark.integration


class StubScreen:
    """Records addstr calls instead of painting them."""

    def __init__(self, height=24, width=80):
        self._size = (height, width)
        self.writes = []          # (row, column, text, attribute)
        self.keys = []
        self.timeouts = []

    def getmaxyx(self):
        return self._size

    def addstr(self, row, column, text, attribute=0):
        self.writes.append((row, column, text, attribute))

    def erase(self): self.writes.clear()
    def refresh(self): pass
    def keypad(self, _): pass
    def timeout(self, value): self.timeouts.append(value)
    def getch(self): return self.keys.pop(0) if self.keys else -1

    def text_at(self, row):
        return "".join(t for r, _, t, _ in self.writes if r == row)

    def row_of(self, needle):
        for row, _, text, _ in self.writes:
            if needle in text:
                return row, text
        raise AssertionError(f"{needle!r} was never drawn")

    def attr_of(self, needle):
        for _, _, text, attribute in self.writes:
            if needle in text:
                return attribute
        raise AssertionError(f"{needle!r} was never drawn")


@pytest.fixture
def screen():
    return StubScreen()


@pytest.fixture
def theme():
    return rt.load_theme(rt.DEFAULT_THEME_PATH)


@pytest.fixture
def view():
    def build(**overrides):
        chunks = overrides.pop("chunks", ["alpha one", "beta two", "gamma three"])
        cursor = overrides.pop("cursor", 1)
        recorded = overrides.pop("recorded", {0})
        base = {
            "title": "script.txt · es · 1/3",
            "chunks": chunks,
            "statuses": rs.chunk_statuses(len(chunks), recorded, cursor),
            "recorded": recorded,
            "cursor": cursor,
            "state": ui.IDLE,
            "tick": 0,
            "elapsed": 0,
            "message": "",
        }
        base.update(overrides)
        return base
    return build


@pytest.fixture
def render(screen, theme):
    """Build the UI once curses is faked out, then draw a view."""
    def run(view_dict, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        widget = ui.RecorderUI(screen, theme)
        widget.draw(view_dict)
        return widget
    return run


class TestLineColouring:
    def test_each_status_draws_with_a_distinct_attribute(
        self, screen, view, render, monkeypatch
    ):
        """Recorded, selected and pending must not look alike."""
        render(view(), monkeypatch)
        attributes = {
            screen.attr_of("alpha one"),
            screen.attr_of("beta two"),
        }
        assert len(attributes) == 2

    def test_selected_line_is_bold(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        assert screen.attr_of("beta two") & curses.A_BOLD

    def test_cursor_marks_the_selected_line(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        _, text = screen.row_of("beta two")
        assert ui.CURSOR_MARK in text

    def test_recorded_line_carries_a_tick(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        _, text = screen.row_of("alpha one")
        assert ui.MARK_RECORDED in text

    def test_pending_line_has_no_tick(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        _, text = screen.row_of("gamma three")
        assert ui.MARK_RECORDED not in text

    def test_line_numbers_are_one_based(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        row, text = screen.row_of("alpha one")
        assert "1" in text


class TestStatusBar:
    def test_idle_bar_shows_the_legend(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        assert "record" in screen.text_at(23)

    def test_idle_bar_reports_idle_state(self, screen, view, render, monkeypatch):
        render(view(), monkeypatch)
        assert "IDLE" in screen.text_at(23)

    def test_recording_bar_reports_recording_state(
        self, screen, view, render, monkeypatch
    ):
        render(view(state=ui.RECORDING), monkeypatch)
        assert "RECORDING" in screen.text_at(23)

    def test_recording_bar_shows_the_elapsed_timer(
        self, screen, view, render, monkeypatch
    ):
        render(view(state=ui.RECORDING, elapsed=64), monkeypatch)
        assert "1:04" in screen.text_at(23)

    def test_dot_alternates_between_consecutive_ticks(
        self, screen, view, render, monkeypatch
    ):
        """The blink is a redraw, so consecutive ticks must differ."""
        widget = render(view(state=ui.RECORDING, tick=0), monkeypatch)
        first = screen.text_at(23)
        widget.draw(view(state=ui.RECORDING, tick=1))
        assert first != screen.text_at(23)

    def test_dot_returns_to_filled_after_two_ticks(
        self, screen, view, render, monkeypatch
    ):
        widget = render(view(state=ui.RECORDING, tick=0), monkeypatch)
        first = screen.text_at(23)
        widget.draw(view(state=ui.RECORDING, tick=2))
        assert first == screen.text_at(23)

    def test_message_is_shown_above_the_bar(self, screen, view, render, monkeypatch):
        render(view(message="clip too short"), monkeypatch)
        assert "clip too short" in screen.text_at(22)


class TestViewportScrolling:
    def test_a_cursor_beyond_the_window_scrolls_into_view(
        self, screen, view, render, monkeypatch
    ):
        chunks = [f"line {n}" for n in range(60)]
        render(view(chunks=chunks, cursor=55, recorded=set()), monkeypatch)
        assert any("line 55" in text for _, _, text, _ in screen.writes)

    def test_offscreen_lines_are_not_drawn(self, screen, view, render, monkeypatch):
        chunks = [f"line {n}" for n in range(60)]
        render(view(chunks=chunks, cursor=55, recorded=set()), monkeypatch)
        assert not any("line 0 " in text for _, _, text, _ in screen.writes)

    def test_long_text_wraps_within_the_window(
        self, screen, view, render, monkeypatch
    ):
        long_chunk = " ".join(["palabra"] * 40)
        render(view(chunks=[long_chunk], cursor=0, recorded=set()), monkeypatch)
        assert all(len(text) < 80 for _, _, text, _ in screen.writes)


class TestKeyMapping:
    @pytest.fixture
    def widget(self, screen, theme, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        return ui.RecorderUI(screen, theme)

    @pytest.mark.parametrize("key,action", [
        (curses.KEY_UP, "up"),
        (curses.KEY_DOWN, "down"),
        (ord(" "), "record"),
        (ord("r"), "redo"),
        (ord("q"), "quit"),
        (ord("s"), "skip"),
        (ord("p"), "play"),
    ])
    def test_keys_map_to_actions(self, widget, screen, key, action):
        screen.keys = [key]
        assert widget.read_key() == action

    def test_timeout_expiry_yields_none(self, widget, screen):
        screen.keys = []
        assert widget.read_key(timeout_ms=500) is None

    def test_timeout_is_passed_to_curses(self, widget, screen):
        widget.read_key(timeout_ms=250)
        assert 250 in screen.timeouts

    def test_no_timeout_blocks(self, widget, screen):
        widget.read_key()
        assert -1 in screen.timeouts

    def test_unknown_key_is_ignored(self, widget, screen):
        screen.keys = [ord("z")]
        assert widget.read_key() is None


class TestConfirm:
    @pytest.fixture
    def widget(self, screen, theme, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        return ui.RecorderUI(screen, theme)

    def test_y_confirms(self, widget, screen):
        screen.keys = [ord("y")]
        assert widget.confirm("Re-record line 5?") is True

    def test_n_declines(self, widget, screen):
        screen.keys = [ord("n")]
        assert widget.confirm("Re-record line 5?") is False

    def test_escape_declines(self, widget, screen):
        """A stray keypress must not destroy a good take."""
        screen.keys = [27]
        assert widget.confirm("Re-record line 5?") is False

    def test_other_keys_are_ignored_until_a_decision(self, widget, screen):
        screen.keys = [ord("x"), ord("j"), ord("y")]
        assert widget.confirm("Re-record line 5?") is True

    def test_question_is_displayed(self, widget, screen):
        screen.keys = [ord("n")]
        widget.confirm("Re-record line 5?")
        assert any("Re-record line 5?" in t for _, _, t, _ in screen.writes)
