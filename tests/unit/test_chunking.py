"""Chunking decides what the user reads aloud, so its bounds must hold exactly."""

import pytest

import whisper_pipeline as wp


def words_in(chunk):
    return len(chunk.split())


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert wp.split_sentences("One two. Three four! Five six?") == [
            "One two.", "Three four!", "Five six?"
        ]

    def test_keeps_decimal_numbers_intact(self):
        assert wp.split_sentences("Costs 3.5 pesos today.") == ["Costs 3.5 pesos today."]

    def test_starts_new_sentence_at_spanish_inverted_mark(self):
        assert wp.split_sentences("Hola amigo. ¿Como estas?") == [
            "Hola amigo.", "¿Como estas?"
        ]

    def test_collapses_irregular_whitespace(self):
        assert wp.split_sentences("  one\n\ttwo   three  ") == ["one two three"]

    def test_returns_empty_list_for_blank_input(self):
        assert wp.split_sentences("   \n  ") == []


class TestParagraphBreaks:
    """A blank line ends a sentence even without terminal punctuation.

    Collapsing all whitespace ran two paragraphs together, so a heading ending
    in ':' merged with the text below it and was then cut mid-clause.
    """

    def test_a_blank_line_separates_sentences(self):
        assert wp.split_sentences("First part:\n\nSecond part.") == [
            "First part:", "Second part."
        ]

    def test_a_single_newline_still_collapses(self):
        """Only a blank line is a break; wrapped prose is one sentence."""
        assert wp.split_sentences("one\ntwo three") == ["one two three"]

    def test_several_blank_lines_are_one_break(self):
        assert wp.split_sentences("A one.\n\n\n\nB two.") == ["A one.", "B two."]

    def test_blank_lines_of_whitespace_still_break(self):
        assert wp.split_sentences("A one:\n   \t \nB two.") == ["A one:", "B two."]

    def test_the_reported_paragraph_is_not_merged(self):
        text = (
            "The following code shows how methods for the boolean and, or, and "
            "xor operations could be expressed using pattern matching syntax:"
            "\n\n"
            "Pattern matching expressions can be simplified by using _ as a "
            "catchall for any value."
        )
        chunks = wp.chunk_text(text)
        assert chunks[0].endswith("syntax:")

    def test_the_second_paragraph_stays_whole(self):
        text = (
            "The following code shows how methods for the boolean and, or, and "
            "xor operations could be expressed using pattern matching syntax:"
            "\n\n"
            "Pattern matching expressions can be simplified by using _ as a "
            "catchall for any value."
        )
        assert wp.chunk_text(text)[1].startswith("Pattern matching expressions")


class TestNaturalBreakSplitting:
    """An overlong sentence is cut at punctuation when one is in range."""

    def test_splits_at_a_comma_rather_than_mid_clause(self):
        text = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu, "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega alef bet gimel."
        )
        assert wp.chunk_text(text)[0].endswith("mu,")

    def test_prefers_a_semicolon_break(self):
        text = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu; "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega alef bet gimel."
        )
        assert wp.chunk_text(text)[0].endswith("mu;")

    def test_falls_back_to_a_word_boundary_without_punctuation(self):
        """No natural break in range: the plain cut is still correct."""
        text = " ".join(f"word{i}" for i in range(40)) + "."
        chunks = wp.chunk_text(text)
        assert words_in(chunks[0]) == wp.MAX_WORDS_PER_CHUNK

    def test_a_break_too_early_is_ignored(self):
        """Cutting at a comma in the first few words would leave a stub line."""
        text = "alpha, " + " ".join(f"word{i}" for i in range(40)) + "."
        assert words_in(wp.chunk_text(text)[0]) > 5

    def test_natural_splits_still_respect_the_maximum(self):
        text = (
            "alpha beta, gamma delta epsilon zeta eta theta iota kappa lambda mu, "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega alef bet."
        )
        assert all(words_in(c) <= wp.MAX_WORDS_PER_CHUNK for c in wp.chunk_text(text))

    def test_every_word_survives_a_natural_split(self):
        text = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu, "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega alef bet gimel."
        )
        assert " ".join(wp.chunk_text(text)).split() == text.split()


class TestChunkText:
    def test_groups_short_sentences_up_to_the_minimum(self):
        """Three-word sentences must be packed together, not recorded one by one."""
        text = " ".join(f"Word{i} is here." for i in range(6))
        chunks = wp.chunk_text(text)
        assert words_in(chunks[0]) >= 10

    def test_never_exceeds_the_maximum_word_count(self):
        text = " ".join(f"word{i}" for i in range(200)) + "."
        assert all(words_in(chunk) <= wp.MAX_WORDS_PER_CHUNK for chunk in wp.chunk_text(text))

    def test_hard_splits_a_single_overlong_sentence(self):
        sentence = " ".join(f"word{i}" for i in range(60)) + "."
        assert len(wp.chunk_text(sentence)) == 3

    def test_preserves_every_word(self):
        text = " ".join(f"word{i}" for i in range(97)) + "."
        rejoined = " ".join(wp.chunk_text(text)).replace(".", "")
        assert rejoined.split() == text.replace(".", "").split()

    def test_returns_empty_list_for_blank_input(self):
        assert wp.chunk_text("") == []

    def test_keeps_short_trailing_remainder(self):
        text = " ".join(f"word{i}" for i in range(12)) + ". Short tail."
        assert wp.chunk_text(text)[-1] == "Short tail."

    @pytest.mark.parametrize("word_count", [1, 9, 10, 25, 26, 51])
    def test_chunks_stay_within_bounds_for_any_length(self, word_count):
        text = " ".join(f"word{i}" for i in range(word_count)) + "."
        chunks = wp.chunk_text(text)
        assert chunks
        assert all(words_in(chunk) <= wp.MAX_WORDS_PER_CHUNK for chunk in chunks)
