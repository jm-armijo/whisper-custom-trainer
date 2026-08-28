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


@pytest.fixture
def render_in_colour(screen, theme):
    """Draw with colour pairs live.

    The plain `render` fixture disables colour, which collapses every style
    onto its bold flag alone - two differently-coloured statuses then compare
    equal, and a colour regression would pass unnoticed.
    """
    def run(view_dict, monkeypatch):
        pairs = {}
        monkeypatch.setattr(curses, "has_colors", lambda: True)
        monkeypatch.setattr(curses, "start_color", lambda: None)
        monkeypatch.setattr(curses, "use_default_colors", lambda: None)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        monkeypatch.setattr(
            curses, "init_pair", lambda index, fg, bg: pairs.__setitem__(index, (fg, bg))
        )
        monkeypatch.setattr(curses, "color_pair", lambda index: index << 8)
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

    def test_a_recorded_selected_line_is_not_drawn_like_a_pending_one(
        self, screen, view, render_in_colour, monkeypatch
    ):
        """Selecting a finished line left it yellow, reading as 'read this next'."""
        render_in_colour(view(cursor=0, recorded={0}), monkeypatch)
        assert screen.attr_of("alpha one") != screen.attr_of("beta two")

    def test_a_recorded_selected_line_shares_the_recorded_colour(
        self, screen, view, render_in_colour, theme, monkeypatch
    ):
        """Green is what says 'done'; only the weight marks the cursor.

        Compared by resolved foreground rather than pair number: each style
        gets its own pair even when two of them ask for the same colour.
        """
        widget = render_in_colour(view(cursor=0, recorded={0, 2}), monkeypatch)
        assert (theme.style(rs.RECORDED_SELECTED).fg
                == theme.style(rs.RECORDED).fg)
        assert widget.attr(rs.RECORDED_SELECTED) != widget.attr(rs.RECORDED)

    def test_a_recorded_selected_line_keeps_the_cursor_mark(
        self, screen, view, render, monkeypatch
    ):
        render(view(cursor=0, recorded={0}), monkeypatch)
        _, text = screen.row_of("alpha one")
        assert ui.CURSOR_MARK in text

    def test_a_recorded_selected_line_is_bold(
        self, screen, view, render, monkeypatch
    ):
        """Bold is what marks the cursor when the colour has gone green."""
        render(view(cursor=0, recorded={0}), monkeypatch)
        assert screen.attr_of("alpha one") & curses.A_BOLD

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


class TestRawEscapeSequences:
    """Some terminals send arrows as raw ESC [ A rather than a KEY_* constant."""

    @pytest.fixture
    def widget(self, screen, theme, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        return ui.RecorderUI(screen, theme)

    @pytest.mark.parametrize("final,action", [
        ("A", "up"), ("B", "down"), ("H", "top"), ("F", "bottom"),
    ])
    def test_raw_arrow_sequence_maps_to_its_action(
        self, widget, screen, final, action
    ):
        screen.keys = [27, ord("["), ord(final)]
        assert widget.read_key() == action

    def test_parameterised_sequence_is_decoded(self, widget, screen):
        """xterm may send ESC [ 1 ; 2 B for a modified arrow."""
        screen.keys = [27, ord("["), ord("1"), ord(";"), ord("2"), ord("B")]
        assert widget.read_key() == "down"

    def test_lone_escape_is_not_an_action(self, widget, screen):
        screen.keys = [27]
        assert widget.read_key() is None

    def test_escape_followed_by_other_text_is_ignored(self, widget, screen):
        screen.keys = [27, ord("x")]
        assert widget.read_key() is None


class TestConfirmOnClosedInput:
    """A -1 read means no key; confirm must not spin on it."""

    @pytest.fixture
    def widget(self, screen, theme, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        return ui.RecorderUI(screen, theme)

    def test_exhausted_input_declines_rather_than_looping(self, widget, screen):
        screen.keys = []          # every getch returns -1
        assert widget.confirm("Re-record line 1?") is False


class TestEscapeRestoresTimeout:
    @pytest.fixture
    def widget(self, screen, theme, monkeypatch):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        return ui.RecorderUI(screen, theme)

    def test_timeout_is_restored_after_an_escape(self, widget, screen):
        """A 50ms escape window must not leak into the next blocking read."""
        screen.keys = [27, ord("x")]
        widget.read_key(timeout_ms=1000)
        assert screen.timeouts[-1] == 1000


class TestZeroWidthScreen:
    def test_a_zero_width_screen_draws_nothing_rather_than_garbage(
        self, theme, view, monkeypatch
    ):
        monkeypatch.setattr(curses, "has_colors", lambda: False)
        monkeypatch.setattr(curses, "curs_set", lambda _: None)
        screen = StubScreen(height=5, width=0)
        widget = ui.RecorderUI(screen, theme)
        widget.draw(view())
        assert all(text == "" for _, _, text, _ in screen.writes)
