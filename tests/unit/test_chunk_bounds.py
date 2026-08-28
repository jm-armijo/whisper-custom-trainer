"""Regression tests for chunk accumulation overflowing the readable maximum."""

import pytest

import whisper_pipeline as wp


def words(count, token="palabra"):
    return " ".join([token] * count) + "."


class TestChunkAccumulation:
    """Packing several sentences must respect MAX_WORDS_PER_CHUNK."""

    @pytest.mark.parametrize(
        "sentence_lengths",
        [(9, 24), (1, 25), (24, 24), (10, 10), (5, 5), (26, 3), (3, 3, 3, 3, 3, 3, 3)],
    )
    def test_no_chunk_exceeds_the_maximum(self, sentence_lengths):
        text = " ".join(words(length) for length in sentence_lengths)

        longest = max(len(chunk.split()) for chunk in wp.chunk_text(text))

        assert longest <= wp.MAX_WORDS_PER_CHUNK

    @pytest.mark.parametrize(
        "sentence_lengths",
        [(9, 24), (1, 25), (24, 24), (26, 3)],
    )
    def test_every_word_survives_chunking(self, sentence_lengths):
        text = " ".join(words(length) for length in sentence_lengths)

        recovered = " ".join(wp.chunk_text(text)).split()

        assert recovered == text.split()
