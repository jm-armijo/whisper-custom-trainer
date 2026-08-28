"""Colour and timing configuration, resolved without a terminal."""

import json

import pytest

import recorder_theme as rt
import whisper_pipeline as wp


@pytest.fixture
def theme_file(tmp_path):
    def build(payload):
        path = tmp_path / "theme.json"
        path.write_text(json.dumps(payload), encoding="utf8")
        return path
    return build


class TestLoadTheme:
    def test_defaults_apply_without_a_file(self, tmp_path):
        theme = rt.load_theme(tmp_path / "absent.json")
        assert theme.style("recorded").fg == "green"

    def test_user_values_override_defaults(self, theme_file):
        theme = rt.load_theme(theme_file({"recorded": {"fg": "cyan"}}))
        assert theme.style("recorded").fg == "cyan"

    def test_unspecified_styles_keep_their_defaults(self, theme_file):
        """A partial file must not blank out the rest of the UI."""
        theme = rt.load_theme(theme_file({"recorded": {"fg": "cyan"}}))
        assert theme.style("selected").fg == "yellow"

    def test_partial_style_keeps_sibling_keys(self, theme_file):
        """Overriding fg alone must not discard the default bold."""
        theme = rt.load_theme(theme_file({"selected": {"fg": "magenta"}}))
        assert theme.style("selected").bold is True

    def test_unknown_colour_name_is_rejected(self, theme_file):
        with pytest.raises(wp.PipelineError, match="chartreuse"):
            rt.load_theme(theme_file({"recorded": {"fg": "chartreuse"}}))

    def test_error_names_the_offending_key(self, theme_file):
        with pytest.raises(wp.PipelineError, match="recorded"):
            rt.load_theme(theme_file({"recorded": {"fg": "chartreuse"}}))

    def test_malformed_json_is_rejected_with_the_path(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf8")
        with pytest.raises(wp.PipelineError, match="broken.json"):
            rt.load_theme(path)

    def test_shipped_default_theme_file_loads(self):
        """The checked-in theme must stay valid."""
        theme = rt.load_theme(rt.DEFAULT_THEME_PATH)
        assert theme.style("selected").fg == "yellow"


class TestRecordedSelectedStyle:
    """A recorded line under the cursor gets its own themeable style rather
    than borrowing one of the other two, so it stays user-configurable."""

    def test_it_is_a_default_style(self, tmp_path):
        assert "recorded_selected" in rt.load_theme(tmp_path / "none.json").names()

    def test_it_is_green_like_a_finished_line(self, tmp_path):
        theme = rt.load_theme(tmp_path / "none.json")
        assert theme.style("recorded_selected").fg == "green"

    def test_it_is_bold_so_the_cursor_stays_visible(self, tmp_path):
        theme = rt.load_theme(tmp_path / "none.json")
        assert theme.style("recorded_selected").bold

    def test_it_differs_from_a_plain_recorded_line(self, tmp_path):
        theme = rt.load_theme(tmp_path / "none.json")
        assert theme.style("recorded_selected") != theme.style("recorded")

    def test_the_user_can_override_it(self, theme_file):
        theme = rt.load_theme(theme_file({"recorded_selected": {"fg": "cyan"}}))
        assert theme.style("recorded_selected").fg == "cyan"


class TestBlinkInterval:
    def test_defaults_to_one_second(self, tmp_path):
        assert rt.load_theme(tmp_path / "absent.json").blink_ms == 1000

    def test_reads_the_configured_interval(self, theme_file):
        assert rt.load_theme(theme_file({"blink_ms": 250})).blink_ms == 250

    def test_rejects_a_non_numeric_interval(self, theme_file):
        with pytest.raises(wp.PipelineError, match="blink_ms"):
            rt.load_theme(theme_file({"blink_ms": "fast"}))

    def test_rejects_a_zero_interval(self, theme_file):
        """Zero would spin the redraw loop at full speed."""
        with pytest.raises(wp.PipelineError, match="blink_ms"):
            rt.load_theme(theme_file({"blink_ms": 0}))

    def test_rejects_a_negative_interval(self, theme_file):
        with pytest.raises(wp.PipelineError, match="blink_ms"):
            rt.load_theme(theme_file({"blink_ms": -5}))


class TestColourResolution:
    def test_resolves_a_named_colour_to_its_curses_constant(self):
        import curses
        assert rt.resolve_colour("green") == curses.COLOR_GREEN

    def test_resolves_an_indexed_colour(self):
        assert rt.resolve_colour("color:214") == 214

    def test_default_sentinel_resolves_to_terminal_default(self):
        assert rt.resolve_colour("default") == -1

    def test_rejects_an_out_of_range_index(self):
        with pytest.raises(wp.PipelineError):
            rt.resolve_colour("color:999")

    def test_rejects_an_unknown_name(self):
        with pytest.raises(wp.PipelineError):
            rt.resolve_colour("chartreuse")


class TestBlinkGlyph:
    def test_alternates_between_filled_and_hollow(self):
        assert rt.blink_glyph(0) != rt.blink_glyph(1)

    def test_repeats_every_two_ticks(self):
        assert rt.blink_glyph(0) == rt.blink_glyph(2)

    def test_starts_filled_so_recording_reads_as_live(self):
        assert rt.blink_glyph(0) == "●"

    def test_hollow_on_the_odd_tick(self):
        assert rt.blink_glyph(1) == "○"


class TestMalformedConfig:
    """A hand-edited config must fail with a clear error, not a traceback."""

    def test_style_given_a_string_instead_of_a_mapping(self, theme_file):
        with pytest.raises(wp.PipelineError, match="recorded"):
            rt.load_theme(theme_file({"recorded": "green"}))

    def test_top_level_array_is_rejected(self, tmp_path):
        path = tmp_path / "arr.json"
        path.write_text("[1, 2, 3]", encoding="utf8")
        with pytest.raises(wp.PipelineError):
            rt.load_theme(path)

    def test_non_string_colour_is_rejected(self, theme_file):
        with pytest.raises(wp.PipelineError, match="pending"):
            rt.load_theme(theme_file({"pending": {"fg": 123}}))

    def test_top_level_string_is_rejected(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text('"hello"', encoding="utf8")
        with pytest.raises(wp.PipelineError):
            rt.load_theme(path)
