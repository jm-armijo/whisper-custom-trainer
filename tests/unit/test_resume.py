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


class TestAppendRow:
    def test_writes_header_once_across_appends(self, csv_path):
        from record_data import append_row

        append_row(csv_path, csv_path.parent / "a.wav", "uno", "es")
        append_row(csv_path, csv_path.parent / "b.wav", "dos", "es")

        header_lines = [
            line for line in csv_path.read_text().splitlines()
            if line.startswith("audio_path")
        ]
        assert len(header_lines) == 1

    def test_appended_rows_are_visible_to_the_resume_count(self, csv_path):
        from record_data import append_row

        append_row(csv_path, csv_path.parent / "a.wav", "uno", "es")
        assert wp.count_recorded_chunks(csv_path, "es") == 1

    def test_preserves_text_containing_commas(self, csv_path):
        import csv

        from record_data import append_row

        append_row(csv_path, csv_path.parent / "a.wav", "Hola, amigo, que tal", "es")

        with csv_path.open(newline="", encoding="utf8") as handle:
            assert next(csv.DictReader(handle))["text"] == "Hola, amigo, que tal"

    def test_preserves_accented_characters(self, csv_path):
        import csv

        from record_data import append_row

        append_row(csv_path, csv_path.parent / "a.wav", "¿Cómo estás? Añejo", "es")

        with csv_path.open(newline="", encoding="utf8") as handle:
            assert next(csv.DictReader(handle))["text"] == "¿Cómo estás? Añejo"
