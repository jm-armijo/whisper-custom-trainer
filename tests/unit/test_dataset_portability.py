"""A dataset recorded on one machine must be usable on another.

The container writes rows under /data/audio; the documented workflow then
rsyncs the clips and the CSV to the laptop, where that directory does not
exist. Storing an absolute path made every one of those rows unreadable to
train.py, and made record_data.py's startup prune delete the lot - opening the
terminal recorder once destroyed a whole recording session.
"""

import csv

import recorder_state as rs
import whisper_pipeline as wp

# Where the container writes, per docker-compose.yml. Nothing under it exists
# on the laptop that later reads the rsynced dataset.
CONTAINER_AUDIO_DIR = "/data/audio"


def read_rows(csv_path):
    with csv_path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def rows_written_in_the_container(csv_path, count=3):
    """Rows exactly as a container recording session leaves them."""
    for index in range(count):
        rs.upsert_row(
            csv_path,
            f"{CONTAINER_AUDIO_DIR}/es_{index:05d}.wav",
            f"line {index}",
            "es",
        )


def clips_rsynced_to(audio_dir, count=3):
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (audio_dir / f"es_{index:05d}.wav").write_bytes(b"RIFF")


class TestAPathWrittenElsewhere:
    def test_the_row_does_not_name_a_directory_from_the_other_machine(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        rows_written_in_the_container(csv_path, count=1)

        assert CONTAINER_AUDIO_DIR not in read_rows(csv_path)[0]["audio_path"]

    def test_the_clip_resolves_against_this_machines_audio_dir(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        rows_written_in_the_container(csv_path, count=1)
        clips_rsynced_to(audio_dir, count=1)

        stored = read_rows(csv_path)[0]["audio_path"]
        assert wp.resolve_audio_path(stored, audio_dir).exists()

    def test_opening_the_recorder_does_not_delete_the_rows(self, tmp_path):
        """record_data.py prunes at startup; it must not read a rsynced row as
        a clip that has gone missing."""
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        rows_written_in_the_container(csv_path)
        clips_rsynced_to(audio_dir)

        rs.prune_missing(csv_path, audio_dir)

        assert len(read_rows(csv_path)) == 3

    def test_the_rows_still_count_as_recorded(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        rows_written_in_the_container(csv_path)
        clips_rsynced_to(audio_dir)

        assert rs.recorded_indices(csv_path, audio_dir, "es") == {0, 1, 2}


class TestRowsAlreadyWrittenAbsolute:
    """Datasets recorded before this change still hold absolute paths; they
    must keep working rather than being silently dropped on the next prune."""

    def test_an_absolute_row_whose_clip_exists_is_kept(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        clips_rsynced_to(audio_dir, count=1)
        _write_legacy_row(csv_path, str(audio_dir / "es_00000.wav"))

        rs.prune_missing(csv_path, audio_dir)

        assert len(read_rows(csv_path)) == 1

    def test_an_absolute_row_whose_clip_is_gone_is_still_pruned(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()
        _write_legacy_row(csv_path, str(audio_dir / "es_00000.wav"))

        rs.prune_missing(csv_path, audio_dir)

        assert read_rows(csv_path) == []

    def test_re_recording_replaces_the_absolute_row_rather_than_adding_one(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        clips_rsynced_to(audio_dir, count=1)
        _write_legacy_row(csv_path, str(audio_dir / "es_00000.wav"))

        rs.upsert_row(csv_path, audio_dir / "es_00000.wav", "nueva toma", "es")

        assert [row["text"] for row in read_rows(csv_path)] == ["nueva toma"]

    def test_train_can_still_read_an_absolute_row(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        clips_rsynced_to(audio_dir, count=1)
        absolute = str(audio_dir / "es_00000.wav")
        _write_legacy_row(csv_path, absolute)

        stored = read_rows(csv_path)[0]["audio_path"]
        assert str(wp.resolve_audio_path(stored, audio_dir)) == absolute


def _write_legacy_row(csv_path, audio_path):
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(wp.CSV_COLUMNS)
        writer.writerow([audio_path, "vieja toma", "es"])
