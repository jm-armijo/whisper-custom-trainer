"""Full-screen recorder view.

Presentation only: this module knows curses, colours and geometry, and nothing
about audio, CSV rows or the filesystem. The controller in record_data.py owns
all of that and hands this class a plain view description to draw.
"""

import curses
import locale
import textwrap

import recorder_state as rs
import recorder_theme as rt

IDLE = "idle"
RECORDING = "recording"

GUTTER = 6          # "  12 ✓" before the text column
MARK_RECORDED = "✓"
CURSOR_MARK = "▸"

KEY_ACTIONS = {
    curses.KEY_UP: "up",
    curses.KEY_DOWN: "down",
    curses.KEY_HOME: "top",
    curses.KEY_END: "bottom",
    ord("k"): "up",
    ord("j"): "down",
    ord(" "): "record",
    ord("\n"): "record",
    ord("r"): "redo",
    ord("p"): "play",
    ord("s"): "skip",
    ord("q"): "quit",
}

LEGEND_IDLE = "↑↓ move  ␣ record  r redo  p play  s skip  q quit"
LEGEND_RECORDING = "␣ stop"


def wrap_chunk(text, width):
    """Wrap one chunk to the text column, never dropping a long word."""
    if not text:
        return [""]
    return textwrap.wrap(
        text, max(width, 1), break_long_words=True, break_on_hyphens=False
    ) or [""]


def viewport_start(cursor, total, height, current):
    """Scroll the window minimally to keep the cursor on screen.

    Holding still whenever the cursor is already visible avoids the viewport
    jumping on every keypress.
    """
    if height <= 0:
        return 0
    start = max(0, min(current, max(total - height, 0)))
    if cursor < start:
        return cursor
    if cursor >= start + height:
        return max(0, cursor - height + 1)
    return start


def elapsed_label(seconds):
    """mm:ss for the recording timer."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class RecorderUI:
    """Draws the script and reads keys. Holds no dataset state of its own."""

    def __init__(self, stdscr, theme):
        self.stdscr = stdscr
        self.theme = theme
        self._top = 0
        self._pairs = {}
        self._init_colours()
        curses.curs_set(0)
        self.stdscr.keypad(True)

    def _init_colours(self):
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass  # A terminal without default-colour support still gets pairs.
        for index, name in enumerate(self.theme.names(), start=1):
            style = self.theme.style(name)
            try:
                curses.init_pair(
                    index, rt.resolve_colour(style.fg), rt.resolve_colour(style.bg)
                )
            except curses.error:
                continue  # Fewer pairs than styles: fall back to plain text.
            self._pairs[name] = index

    def attr(self, name):
        """Curses attribute for a style, degrading to bold-only without colour."""
        style = self.theme.style(name)
        attribute = curses.A_BOLD if style.bold else curses.A_NORMAL
        if name in self._pairs:
            attribute |= curses.color_pair(self._pairs[name])
        return attribute

    def draw(self, view):
        """Render a view: title, lines, cursor, state, tick, message."""
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        body_height = max(height - 2, 1)

        self._draw_title(view, width)
        self._draw_lines(view, body_height, width)
        self._draw_status(view, height, width)
        self.stdscr.refresh()

    def _draw_title(self, view, width):
        self._put(0, 0, view["title"].ljust(width - 1)[: width - 1], self.attr("border"))

    def _draw_lines(self, view, body_height, width):
        statuses = view["statuses"]
        chunks = view["chunks"]
        text_width = max(width - GUTTER - 1, 8)

        # Rows are laid out from the viewport top; a chunk may wrap to several.
        self._top = viewport_start(view["cursor"], len(chunks), body_height, self._top)
        row = 1
        for index in range(self._top, len(chunks)):
            if row > body_height:
                break
            status = statuses[index]
            attribute = self.attr(status)
            mark = CURSOR_MARK if index == view["cursor"] else " "
            tick = MARK_RECORDED if index in view["recorded"] else " "

            for offset, piece in enumerate(wrap_chunk(chunks[index], text_width)):
                if row > body_height:
                    break
                prefix = (
                    f"{mark}{index + 1:>3} {tick} " if offset == 0
                    else " " * GUTTER
                )
                self._put(row, 0, (prefix + piece)[: width - 1], attribute)
                row += 1

    def _draw_status(self, view, height, width):
        recording = view["state"] == RECORDING
        style = "status_recording" if recording else "status_idle"
        attribute = self.attr(style)

        if recording:
            label = f" RECORDING {rt.blink_glyph(view['tick'])} │ "
            label += f"{elapsed_label(view['elapsed'])}  {LEGEND_RECORDING}"
        else:
            label = f" {view['state'].upper()} │ {LEGEND_IDLE}"

        self._put(height - 1, 0, label.ljust(width - 1)[: width - 1], attribute)

        message = view.get("message") or ""
        if message:
            self._put(height - 2, 0, message[: width - 1], self.attr("message"))

    def _put(self, row, column, text, attribute):
        """addstr past the last cell raises; the final cell is never writable."""
        try:
            self.stdscr.addstr(row, column, text, attribute)
        except curses.error:
            pass

    def read_key(self, timeout_ms=None):
        """Map a keypress to an action. Returns None when the timeout expires."""
        self.stdscr.timeout(-1 if timeout_ms is None else int(timeout_ms))
        key = self.stdscr.getch()
        if key == -1:
            return None
        if key == curses.KEY_RESIZE:
            return "resize"
        return KEY_ACTIONS.get(key)

    def confirm(self, question):
        """Blocking y/n prompt in the status bar."""
        height, width = self.stdscr.getmaxyx()
        self._put(
            height - 1, 0, f" {question} ".ljust(width - 1)[: width - 1],
            self.attr("status_recording"),
        )
        self.stdscr.refresh()
        self.stdscr.timeout(-1)
        while True:
            key = self.stdscr.getch()
            if key in (ord("y"), ord("Y")):
                return True
            if key in (ord("n"), ord("N"), 27, ord("q")):
                return False


def start(main, *args):
    """Enter curses with the locale set, restoring the terminal on any exit.

    setlocale is required before initscr or accented text and the dot glyphs
    render as replacement characters.
    """
    locale.setlocale(locale.LC_ALL, "")
    return curses.wrapper(main, *args)
