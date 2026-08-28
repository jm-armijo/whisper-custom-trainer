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
