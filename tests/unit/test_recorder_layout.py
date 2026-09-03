"""Wrapping and scrolling are pure geometry; no terminal involved."""

import recorder_ui as ui


class TestWrapChunk:
    def test_short_text_stays_on_one_line(self):
        assert ui.wrap_chunk("uno dos", 40) == ["uno dos"]

    def test_long_text_splits_at_word_boundaries(self):
        lines = ui.wrap_chunk("alpha beta gamma delta", 12)
        assert all(len(line) <= 12 for line in lines)

    def test_no_word_is_lost_or_duplicated(self):
        text = "alpha beta gamma delta epsilon zeta"
        assert " ".join(ui.wrap_chunk(text, 14)).split() == text.split()

    def test_a_word_longer_than_the_width_is_not_dropped(self):
        assert "".join(ui.wrap_chunk("supercalifragilistic", 8)) == "supercalifragilistic"

    def test_accented_text_survives_wrapping(self):
        text = "¿Cómo estás amigo? Añejo pequeño"
        assert " ".join(ui.wrap_chunk(text, 12)).split() == text.split()

    def test_empty_text_yields_one_empty_line(self):
        assert ui.wrap_chunk("", 10) == [""]

    def test_a_tiny_width_still_returns_lines(self):
        assert ui.wrap_chunk("alpha beta", 1) != []


class TestViewportStart:
    def test_starts_at_zero_when_everything_fits(self):
        assert ui.viewport_start(cursor=2, total=5, height=10, current=0) == 0

    def test_scrolls_down_to_keep_the_cursor_visible(self):
        assert ui.viewport_start(cursor=9, total=20, height=5, current=0) == 5

    def test_scrolls_up_when_the_cursor_moves_above(self):
        assert ui.viewport_start(cursor=2, total=20, height=5, current=8) == 2

    def test_holds_still_while_the_cursor_stays_in_view(self):
        """No jitter: an in-view cursor must not move the viewport."""
        assert ui.viewport_start(cursor=6, total=20, height=5, current=5) == 5

    def test_never_scrolls_past_the_end(self):
        assert ui.viewport_start(cursor=19, total=20, height=5, current=0) == 15

    def test_never_returns_a_negative_start(self):
        assert ui.viewport_start(cursor=0, total=3, height=5, current=0) == 0


class TestElapsedLabel:
    def test_formats_seconds_as_minutes_and_seconds(self):
        assert ui.elapsed_label(64) == "1:04"

    def test_pads_seconds_below_ten(self):
        assert ui.elapsed_label(5) == "0:05"

    def test_handles_zero(self):
        assert ui.elapsed_label(0) == "0:00"

    def test_handles_over_ten_minutes(self):
        assert ui.elapsed_label(605) == "10:05"
