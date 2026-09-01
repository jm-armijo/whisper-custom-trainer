"""Concurrent writers must not lose dataset rows.

The web recorder serves requests on a ThreadingHTTPServer, and the terminal
recorder can be writing the same dataset.csv from a separate process at the
same time. Every one of those writers does read-modify-write over the whole
file, so without a lock the last writer wins and every other take is silently
dropped - having already told its client the save succeeded.
"""

import csv
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import recorder_state as rs
import whisper_pipeline as wp

WRITERS = 40


def read_rows(csv_path):
    with csv_path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


class TestConcurrentUpsert:
    def test_every_concurrent_take_survives(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        def save(index):
            rs.upsert_row(
                csv_path, audio_dir / f"es_{index:05d}.wav", f"line {index}", "es"
            )

        with ThreadPoolExecutor(max_workers=WRITERS) as pool:
            list(pool.map(save, range(WRITERS)))

        assert len(read_rows(csv_path)) == WRITERS

    def test_no_row_is_left_half_written(self, tmp_path):
        """A reader that opens the file mid-write must never see a torn row."""
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        def save(index):
            rs.upsert_row(
                csv_path, audio_dir / f"es_{index:05d}.wav", f"line {index}", "es"
            )

        with ThreadPoolExecutor(max_workers=WRITERS) as pool:
            list(pool.map(save, range(WRITERS)))

        texts = {row["text"] for row in read_rows(csv_path)}
        assert texts == {f"line {index}" for index in range(WRITERS)}


class TestConcurrentPrune:
    def test_a_prune_racing_saves_keeps_the_saved_rows(self, tmp_path):
        """delete_chunk rewrites the whole CSV; a save landing mid-rewrite must
        not be erased by it."""
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        def work(index):
            wav = audio_dir / f"es_{index:05d}.wav"
            wav.write_bytes(b"RIFF")
            rs.upsert_row(csv_path, wav, f"line {index}", "es")
            rs.prune_missing(csv_path, audio_dir)

        with ThreadPoolExecutor(max_workers=WRITERS) as pool:
            list(pool.map(work, range(WRITERS)))

        assert len(read_rows(csv_path)) == WRITERS


# The TUI and the web server are separate processes writing one dataset.csv, so
# an in-process threading.Lock would not serialise them; only a filesystem lock
# does. Subprocesses are the only way to prove that.
WRITER_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from pathlib import Path
    import recorder_state as rs

    csv_path, audio_dir = sys.argv[1], Path(sys.argv[2])
    first, count = int(sys.argv[3]), int(sys.argv[4])
    for offset in range(count):
        index = first + offset
        rs.upsert_row(csv_path, audio_dir / f"es_{{index:05d}}.wav", f"line {{index}}", "es")
    """
)


class TestSeparateProcesses:
    def test_two_processes_writing_at_once_lose_nothing(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        script = tmp_path / "writer.py"
        script.write_text(
            WRITER_SCRIPT.format(root=str(wp.PROJECT_ROOT)), encoding="utf8"
        )

        per_process = 15
        processes = [
            subprocess.Popen([
                sys.executable, str(script), str(csv_path), str(audio_dir),
                str(start), str(per_process),
            ])
            for start in (0, per_process, 2 * per_process)
        ]
        for process in processes:
            assert process.wait(timeout=60) == 0

        assert len(read_rows(csv_path)) == 3 * per_process
