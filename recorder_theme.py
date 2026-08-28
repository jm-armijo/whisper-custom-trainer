"""User-editable colours and timing for the recorder UI.

Kept apart from recorder_ui so the configuration can be parsed and validated
without a terminal, which is also what makes it directly testable.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import whisper_pipeline as wp

DEFAULT_THEME_PATH = wp.PROJECT_ROOT / "recorder_theme.json"

# Blinking is driven by redrawing on a timer rather than curses.A_BLINK, which
# most modern terminals (macOS Terminal, iTerm2) silently ignore.
FILLED_DOT = "●"
HOLLOW_DOT = "○"

DEFAULTS = {
    "recorded": {"fg": "green", "bold": False},
    "selected": {"fg": "yellow", "bold": True},
    # Green says 'done'; the bold weight (and the cursor mark) is what shows
    # the line is also selected, so it still reads right without colour.
    "recorded_selected": {"fg": "green", "bold": True},
    "pending": {"fg": "white", "bold": False},
    "status_idle": {"fg": "black", "bg": "white", "bold": False},
    "status_recording": {"fg": "black", "bg": "red", "bold": True},
    "record_dot": {"fg": "red", "bold": True},
    "border": {"fg": "blue", "bold": False},
    "message": {"fg": "yellow", "bold": False},
}

DEFAULT_BLINK_MS = 1000

COLOUR_NAMES = (
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
)


@dataclass(frozen=True)
class Style:
    fg: str
    bg: str = "default"
    bold: bool = False


class Theme:
    def __init__(self, styles, blink_ms):
        self._styles = styles
        self.blink_ms = blink_ms

    def style(self, name):
        return self._styles[name]

    def names(self):
        return tuple(self._styles)


def resolve_colour(value):
    """Map a config colour to a curses colour number.

    'default' is -1, the terminal's own colour, which use_default_colors enables.
    """
    import curses

    if value == "default":
        return -1
    if value.startswith("color:"):
        try:
            index = int(value.removeprefix("color:"))
        except ValueError:
            raise wp.PipelineError(f"Invalid indexed colour: {value!r}") from None
        if not 0 <= index <= 255:
            raise wp.PipelineError(f"Colour index out of range 0-255: {value!r}")
        return index
    if value in COLOUR_NAMES:
        return getattr(curses, f"COLOR_{value.upper()}")
    raise wp.PipelineError(
        f"Unknown colour {value!r}. Use one of {', '.join(COLOUR_NAMES)}, "
        "'default', or 'color:N' for a 256-colour index."
    )


def blink_glyph(tick):
    """Filled on even ticks so a fresh recording starts visibly lit."""
    return FILLED_DOT if tick % 2 == 0 else HOLLOW_DOT


def load_theme(path=None):
    """Merge a JSON config over the defaults, validating every value.

    A bad colour name raises rather than falling back silently: a typo that
    quietly renders the wrong colour is harder to notice than a startup error.
    """
    source = Path(path) if path is not None else DEFAULT_THEME_PATH
    payload = _read_payload(source)
    if not isinstance(payload, dict):
        raise wp.PipelineError(
            f"{source} must contain a JSON object, got {type(payload).__name__}"
        )

    blink_ms = _validate_blink(payload.pop("blink_ms", DEFAULT_BLINK_MS))

    styles = {}
    for name, defaults in DEFAULTS.items():
        override = payload.get(name, {})
        if not isinstance(override, dict):
            raise wp.PipelineError(
                f"{name} must be an object like "
                f'{{"fg": "green"}}, got {type(override).__name__}'
            )
        merged = {**defaults, **override}
        for key in ("fg", "bg"):
            if key in merged:
                if not isinstance(merged[key], str):
                    raise wp.PipelineError(
                        f"{name}.{key} must be a colour name string, "
                        f"got {merged[key]!r}"
                    )
                _validate_colour(name, merged[key])
        styles[name] = Style(
            fg=merged.get("fg", "default"),
            bg=merged.get("bg", "default"),
            bold=bool(merged.get("bold", False)),
        )

    return Theme(styles, blink_ms)


def _read_payload(source):
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise wp.PipelineError(f"{source} is not valid JSON: {error}") from None


def _validate_colour(style_name, value):
    try:
        resolve_colour(value)
    except wp.PipelineError as error:
        raise wp.PipelineError(f"{style_name}: {error}") from None


def _validate_blink(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise wp.PipelineError(f"blink_ms must be a number, got {value!r}")
    if value <= 0:
        raise wp.PipelineError(f"blink_ms must be greater than zero, got {value!r}")
    return int(value)
