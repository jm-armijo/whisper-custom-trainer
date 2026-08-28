"""Chunk status derives from the CSV and the wav files that actually exist."""

import csv

import pytest

import recorder_state as rs
import whisper_pipeline as wp


@pytest.fixture
def dataset(tmp_path):
    """Build a dataset.csv plus the wav files that back it.

    rows: (index, language, wav_exists)
    """
    audio_dir = tmp_path / "data"
    audio_dir.mkdir()
    csv_path = tmp_path / "dataset.csv"

    def build(rows):
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(wp.CSV_COLUMNS)
            for index, language, exists in rows:
                wav = audio_dir / f"{language}_{index:05d}.wav"
                writer.writerow([str(wav), f"text {index}", language])
                if exists:
                    wav.write_bytes(b"RIFF")
        return csv_path, audio_dir

    return build


class TestRecordedIndices:
    def test_empty_without_a_dataset(self, tmp_path):
        assert rs.recorded_indices(tmp_path / "none.csv", tmp_path, "es") == set()

    def test_collects_indices_whose_wav_exists(self, dataset):
        csv_path, audio_dir = dataset([(0, "es", True), (1, "es", True)])
        assert rs.recorded_indices(csv_path, audio_dir, "es") == {0, 1}

    def test_ignores_a_row_whose_wav_was_deleted(self, dataset):
        """Deleting a clip must re-open that line for recording."""
        csv_path, audio_dir = dataset([(0, "es", True), (1, "es", False)])
        assert rs.recorded_indices(csv_path, audio_dir, "es") == {0}

    def test_ignores_the_other_language(self, dataset):
        csv_path, audio_dir = dataset([(0, "en", True), (1, "es", True)])
        assert rs.recorded_indices(csv_path, audio_dir, "es") == {1}

    def test_keeps_a_gap_left_by_a_skipped_chunk(self, dataset):
        csv_path, audio_dir = dataset([(0, "es", True), (2, "es", True)])
        assert rs.recorded_indices(csv_path, audio_dir, "es") == {0, 2}


class TestFirstUnrecorded:
    def test_returns_the_gap_not_the_maximum_plus_one(self):
        """The old resume skipped past gaps forever; this must not."""
        assert rs.first_unrecorded(5, {0, 2}) == 1

    def test_returns_zero_when_nothing_recorded(self):
        assert rs.first_unrecorded(3, set()) == 0

    def test_clamps_to_the_last_line_when_all_recorded(self):
        assert rs.first_unrecorded(3, {0, 1, 2}) == 2

    def test_handles_an_empty_script(self):
        assert rs.first_unrecorded(0, set()) == 0


class TestChunkStatuses:
    def test_marks_cursor_selected_over_pending(self):
        assert rs.chunk_statuses(3, set(), 1) == [rs.PENDING, rs.SELECTED, rs.PENDING]

    def test_cursor_on_a_recorded_line_still_reads_selected(self):
        """The cursor must be visible even on work already done."""
        assert rs.chunk_statuses(2, {0}, 0) == [rs.SELECTED, rs.PENDING]

    def test_marks_recorded_lines(self):
        assert rs.chunk_statuses(3, {0, 2}, 1) == [
            rs.RECORDED, rs.SELECTED, rs.RECORDED,
        ]


class TestUpsertRow:
    def test_appends_a_new_row_with_a_header(self, tmp_path):
        path = tmp_path / "dataset.csv"
        rs.upsert_row(path, tmp_path / "es_00000.wav", "uno", "es")

        rows = list(csv.DictReader(path.open(newline="", encoding="utf8")))
        assert [r["text"] for r in rows] == ["uno"]

    def test_replaces_rather_than_duplicates_on_re_record(self, tmp_path):
        path = tmp_path / "dataset.csv"
        wav = tmp_path / "es_00000.wav"
        rs.upsert_row(path, wav, "primera toma", "es")
        rs.upsert_row(path, wav, "segunda toma", "es")

        rows = list(csv.DictReader(path.open(newline="", encoding="utf8")))
        assert [r["text"] for r in rows] == ["segunda toma"]

    def test_preserves_row_order_when_replacing(self, tmp_path):
        path = tmp_path / "dataset.csv"
        for index in range(3):
            rs.upsert_row(path, tmp_path / f"es_{index:05d}.wav", f"t{index}", "es")

        rs.upsert_row(path, tmp_path / "es_00001.wav", "rehecho", "es")

        rows = list(csv.DictReader(path.open(newline="", encoding="utf8")))
        assert [r["text"] for r in rows] == ["t0", "rehecho", "t2"]

    def test_writes_the_header_once(self, tmp_path):
        path = tmp_path / "dataset.csv"
        rs.upsert_row(path, tmp_path / "a.wav", "uno", "es")
        rs.upsert_row(path, tmp_path / "b.wav", "dos", "es")

        headers = [l for l in path.read_text().splitlines() if l.startswith("audio_path")]
        assert len(headers) == 1

    def test_preserves_commas_and_accents(self, tmp_path):
        path = tmp_path / "dataset.csv"
        rs.upsert_row(path, tmp_path / "a.wav", "¿Cómo estás, amigo?", "es")

        rows = list(csv.DictReader(path.open(newline="", encoding="utf8")))
        assert rows[0]["text"] == "¿Cómo estás, amigo?"
