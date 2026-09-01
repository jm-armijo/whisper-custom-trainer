"""Which scripts are available to read, and how far each one has got.

Pure domain: every rule here is asserted without a server, a socket or a
request object, which is what proves the web UI could be replaced wholesale
without touching these functions.
"""

import csv

import pytest

import recorder_scripts as rsc
import whisper_pipeline as wp


@pytest.fixture
def scripts_dir(tmp_path):
    directory = tmp_path / "scripts"
    directory.mkdir()

    def build(files):
        for name, text in files.items():
            (directory / name).write_text(text, encoding="utf8")
        return directory

    return build


# Long enough that each sentence becomes its own chunk: chunk_text merges
# consecutive short sentences until MIN_WORDS_PER_CHUNK is reached.
LINE = "this is a deliberately long sentence with plenty of words in it number {n}."


def script_of(sentences):
    return "\n\n".join(LINE.format(n=n) for n in range(sentences))


@pytest.fixture
def dataset(tmp_path):
    """A dataset.csv plus the wavs backing it, as recorder_state expects."""
    audio_dir = tmp_path / "data"
    audio_dir.mkdir()
    csv_path = tmp_path / "dataset.csv"

    def build(rows):
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(wp.CSV_COLUMNS)
            for index, language in rows:
                wav = audio_dir / f"{language}_{index:05d}.wav"
                writer.writerow([str(wav), f"text {index}", language])
                wav.write_bytes(b"RIFF")
        return csv_path, audio_dir

    return build


class TestInferLanguage:
    """scripts/es.txt is the existing convention; the stem carries the label."""

    def test_reads_the_language_from_the_stem(self):
        assert rsc.infer_language("scripts/es.txt") == "es"

    def test_recognises_every_supported_language(self):
        for language in wp.SUPPORTED_LANGUAGES:
            assert rsc.infer_language(f"scripts/{language}.txt") == language

    def test_is_case_insensitive(self):
        assert rsc.infer_language("scripts/ES.txt") == "es"

    def test_returns_none_for_an_unrecognised_stem(self):
        """An arbitrary name cannot be guessed, so the caller must supply one."""
        assert rsc.infer_language("scripts/chapter-one.txt") is None

    def test_does_not_match_a_language_buried_in_the_stem(self):
        """'expenses' contains 'es' but is not a Spanish script."""
        assert rsc.infer_language("scripts/expenses.txt") is None


class TestResolveLanguage:
    def test_prefers_the_explicit_language_over_the_stem(self):
        assert rsc.resolve_language("scripts/es.txt", "en") == "en"

    def test_falls_back_to_the_stem(self):
        assert rsc.resolve_language("scripts/es.txt", None) == "es"

    def test_rejects_an_unguessable_script_without_a_language(self):
        with pytest.raises(wp.PipelineError, match="language"):
            rsc.resolve_language("scripts/chapter-one.txt", None)

    def test_rejects_an_unsupported_explicit_language(self):
        with pytest.raises(wp.PipelineError, match="fr"):
            rsc.resolve_language("scripts/es.txt", "fr")


class TestListScripts:
    def test_lists_only_txt_files(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola", "notes.md": "x", "en.txt": "hi"})
        assert [item["name"] for item in rsc.list_scripts(directory)] == [
            "en.txt", "es.txt",
        ]

    def test_sorts_by_name_so_the_order_is_stable(self, scripts_dir):
        directory = scripts_dir({"c.txt": "x", "a.txt": "x", "b.txt": "x"})
        assert [item["name"] for item in rsc.list_scripts(directory)] == [
            "a.txt", "b.txt", "c.txt",
        ]

    def test_carries_the_path_so_the_caller_need_not_rebuild_it(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        assert rsc.list_scripts(directory)[0]["path"] == directory / "es.txt"

    def test_carries_the_inferred_language(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola", "misc.txt": "x"})
        languages = {item["name"]: item["language"] for item in rsc.list_scripts(directory)}
        assert languages == {"es.txt": "es", "misc.txt": None}

    def test_a_missing_directory_lists_nothing(self, tmp_path):
        assert rsc.list_scripts(tmp_path / "absent") == []


class TestScriptProgress:
    def test_chunks_the_script_text(self, scripts_dir, dataset):
        directory = scripts_dir({"es.txt": " ".join(f"w{n}" for n in range(30))})
        csv_path, audio_dir = dataset([])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["chunks"] == wp.chunk_text(" ".join(f"w{n}" for n in range(30)))

    def test_counts_nothing_recorded_on_a_fresh_dataset(self, scripts_dir, dataset):
        directory = scripts_dir({"es.txt": "hola mundo"})
        csv_path, audio_dir = dataset([])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert (progress["recorded"], progress["recorded_count"]) == (set(), 0)

    def test_reports_the_indices_already_recorded(self, scripts_dir, dataset):
        text = script_of(6)
        directory = scripts_dir({"es.txt": text})
        csv_path, audio_dir = dataset([(0, "es"), (2, "es")])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["recorded"] == {0, 2}

    def test_ignores_takes_recorded_in_the_other_language(self, scripts_dir, dataset):
        text = script_of(4)
        directory = scripts_dir({"es.txt": text})
        csv_path, audio_dir = dataset([(0, "en")])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["recorded"] == set()

    def test_reports_the_total_so_the_caller_need_not_count(self, scripts_dir, dataset):
        text = script_of(5)
        directory = scripts_dir({"es.txt": text})
        csv_path, audio_dir = dataset([])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["total"] == len(progress["chunks"])

    def test_is_complete_only_when_every_chunk_is_recorded(self, scripts_dir, dataset):
        text = script_of(3)
        directory = scripts_dir({"es.txt": text})
        chunks = wp.chunk_text(text)
        csv_path, audio_dir = dataset([(index, "es") for index in range(len(chunks))])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["complete"] is True

    def test_is_incomplete_with_a_gap(self, scripts_dir, dataset):
        text = script_of(3)
        directory = scripts_dir({"es.txt": text})
        csv_path, audio_dir = dataset([(0, "es")])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["complete"] is False

    def test_an_empty_script_is_not_reported_complete(self, scripts_dir, dataset):
        """Zero of zero is vacuously 'all recorded'; calling it done would
        hide a script whose text failed to chunk."""
        directory = scripts_dir({"es.txt": "   \n\n  "})
        csv_path, audio_dir = dataset([])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert (progress["total"], progress["complete"]) == (0, False)

    def test_points_at_the_first_gap_rather_than_the_end(self, scripts_dir, dataset):
        text = script_of(5)
        directory = scripts_dir({"es.txt": text})
        csv_path, audio_dir = dataset([(0, "es"), (2, "es")])

        progress = rsc.script_progress(directory / "es.txt", csv_path, audio_dir, "es")

        assert progress["next_index"] == 1

    def test_rejects_a_missing_script(self, tmp_path, dataset):
        csv_path, audio_dir = dataset([])
        with pytest.raises(wp.PipelineError, match="not found"):
            rsc.script_progress(tmp_path / "absent.txt", csv_path, audio_dir, "es")


class TestChunkView:
    """Per-chunk state for a list view, as data rather than colours."""

    def test_pairs_each_chunk_with_its_recorded_flag(self):
        assert rsc.chunk_view(["uno", "dos"], {1}) == [
            {"index": 0, "text": "uno", "recorded": False},
            {"index": 1, "text": "dos", "recorded": True},
        ]

    def test_is_empty_for_an_empty_script(self):
        assert rsc.chunk_view([], set()) == []


class TestResolveScript:
    """The server binds to a LAN address, so a name must not escape the dir."""

    def test_resolves_a_plain_name(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        assert rsc.resolve_script(directory, "es.txt") == directory / "es.txt"

    def test_rejects_a_parent_traversal(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "../secrets.txt")

    def test_rejects_a_deep_traversal(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "../../etc/passwd")

    def test_rejects_an_absolute_path(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "/etc/passwd")

    def test_rejects_a_nested_subdirectory(self, scripts_dir):
        """Scripts are a flat directory; a separator is always an escape attempt."""
        directory = scripts_dir({"es.txt": "hola"})
        (directory / "sub").mkdir()
        (directory / "sub" / "inner.txt").write_text("x", encoding="utf8")

        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "sub/inner.txt")

    def test_rejects_a_symlink_leaving_the_directory(self, scripts_dir, tmp_path):
        """A resolved path outside the dir is an escape however it was built."""
        directory = scripts_dir({"es.txt": "hola"})
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf8")
        (directory / "link.txt").symlink_to(outside)

        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "link.txt")

    def test_rejects_a_missing_script(self, scripts_dir):
        directory = scripts_dir({"es.txt": "hola"})
        with pytest.raises(wp.PipelineError, match="not found"):
            rsc.resolve_script(directory, "absent.txt")

    def test_rejects_a_non_txt_file(self, scripts_dir):
        """Only the readable material is addressable, not any file that landed
        in the directory."""
        directory = scripts_dir({"es.txt": "hola", "notes.md": "x"})
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "notes.md")
