"""Resume support prevents re-recording work after an interrupted session."""

import whisper_pipeline as wp


class TestCountRecordedChunks:
    def test_returns_zero_when_dataset_is_absent(self, tmp_path):
        assert wp.count_recorded_chunks(tmp_path / "missing.csv", "es") == 0

    def test_counts_only_the_requested_language(self, write_csv):
        path = write_csv([("uno", "es"), ("one", "en"), ("dos", "es")])
        assert wp.count_recorded_chunks(path, "es") == 2

    def test_ignores_a_language_with_no_rows(self, write_csv):
        path = write_csv([("uno", "es")])
        assert wp.count_recorded_chunks(path, "en") == 0

    def test_counts_header_only_file_as_empty(self, write_csv):
        assert wp.count_recorded_chunks(write_csv([]), "es") == 0
