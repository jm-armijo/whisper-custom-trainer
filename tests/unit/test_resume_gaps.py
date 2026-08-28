"""Regression tests for resuming after a skipped chunk."""

import csv

import pytest

import whisper_pipeline as wp


@pytest.fixture
def write_rows(csv_path):
    """Write dataset rows verbatim, so filenames may skip an index."""

    def build(rows):
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(wp.CSV_COLUMNS)
            writer.writerows(rows)
        return csv_path

    return build


class TestNextChunkIndex:
    """Resume position comes from recorded filenames, not the row count."""

    def test_starts_at_zero_without_a_dataset(self, tmp_path):
        assert wp.next_chunk_index(tmp_path / "missing.csv", "es") == 0

    def test_skips_past_a_gap_left_by_a_skipped_chunk(self, write_rows, csv_path):
        write_rows(
            [
                ["data/es_00000.wav", "uno", "es"],
                ["data/es_00001.wav", "dos", "es"],
                ["data/es_00003.wav", "cuatro", "es"],
            ]
        )

        assert wp.next_chunk_index(csv_path, "es") == 4

    def test_counts_only_the_requested_language(self, write_rows, csv_path):
        write_rows(
            [
                ["data/en_00000.wav", "one", "en"],
                ["data/en_00001.wav", "two", "en"],
                ["data/es_00000.wav", "uno", "es"],
            ]
        )

        assert wp.next_chunk_index(csv_path, "es") == 1
