"""Which scripts are available to read, and how far each one has got.

Pure domain: every rule here is asserted without a server, a socket or a
request object, which is what proves the web UI could be replaced wholesale
without touching these functions.
"""

import csv

import pytest

import recorder_scripts as rsc
import recorder_state as rs
import whisper_pipeline as wp


@pytest.fixture
def scripts_dir(tmp_path):
    directory = tmp_path / "scripts"
    directory.mkdir()

    def build(files):
        for name, text in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf8")
        return directory

    return build


# Long enough that each sentence becomes its own chunk: chunk_text merges
# consecutive short sentences until MIN_WORDS_PER_CHUNK is reached.
LINE = "this is a deliberately long sentence with plenty of words in it number {n}."


def script_of(sentences):
    return "\n\n".join(LINE.format(n=n) for n in range(sentences))


@pytest.fixture
def dataset(tmp_path):
    """A dataset.csv plus the wavs backing it, as recorder_state expects.

    Clips are named through rs.clip_path rather than by hand, so the fixture
    cannot drift from the scheme the two recorders actually write - which is
    how the cross-script collision stayed invisible to these tests.
    `script` is the qualified name the take was read from.
    """
    audio_dir = tmp_path / "data"
    audio_dir.mkdir()
    csv_path = tmp_path / "dataset.csv"

    def build(rows, script="es.txt"):
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(wp.CSV_COLUMNS)
            for index, language in rows:
                wav = rs.clip_path(audio_dir, language, index, script)
                writer.writerow([str(wav), f"text {index}", language])
                wav.write_bytes(b"RIFF")
        return csv_path, audio_dir

    return build


class TestListScripts:
    def test_lists_only_txt_files(self, scripts_dir):
        directory = scripts_dir(
            {"es/a.txt": "hola", "es/notes.md": "x", "en/a.txt": "hi"}
        )
        assert [item["name"] for item in rsc.list_scripts(directory)] == [
            "en/a.txt", "es/a.txt",
        ]

    def test_sorts_by_name_so_the_order_is_stable(self, scripts_dir):
        directory = scripts_dir({"en/c.txt": "x", "en/a.txt": "x", "en/b.txt": "x"})
        assert [item["name"] for item in rsc.list_scripts(directory)] == [
            "en/a.txt", "en/b.txt", "en/c.txt",
        ]

    def test_carries_the_path_so_the_caller_need_not_rebuild_it(self, scripts_dir):
        directory = scripts_dir({"es/a.txt": "hola"})
        assert rsc.list_scripts(directory)[0]["path"] == directory / "es" / "a.txt"

    def test_carries_the_language_its_directory_names(self, scripts_dir):
        directory = scripts_dir({"es/a.txt": "hola", "en/b.txt": "hi"})
        languages = {item["name"]: item["language"] for item in rsc.list_scripts(directory)}
        assert languages == {"es/a.txt": "es", "en/b.txt": "en"}

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
        directory = scripts_dir({"es/a.txt": "hola"})
        assert rsc.resolve_script(directory, "es/a.txt") == directory / "es" / "a.txt"

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
        directory = scripts_dir({"es/a.txt": "hola"})
        with pytest.raises(wp.PipelineError, match="not found"):
            rsc.resolve_script(directory, "es/absent.txt")

    def test_rejects_a_non_txt_file(self, scripts_dir):
        """Only the readable material is addressable, not any file that landed
        in the directory."""
        directory = scripts_dir({"es.txt": "hola", "notes.md": "x"})
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            rsc.resolve_script(directory, "notes.md")


class TestLanguageDirectories:
    """One directory per language: the path states the language, so nothing is
    inferred from a filename and an unlabelled file cannot exist."""

    @pytest.fixture
    def tree(self, tmp_path):
        for language, names in (("en", ("general.txt", "software.txt")),
                                ("es", ("phonetics.txt",))):
            folder = tmp_path / language
            folder.mkdir()
            for name in names:
                (folder / name).write_text("A sentence to read aloud.", encoding="utf8")
        return tmp_path

    def test_lists_every_language_directory(self, tree):
        names = {row["name"] for row in rsc.list_scripts(tree)}
        assert names == {"en/general.txt", "en/software.txt", "es/phonetics.txt"}

    def test_the_directory_names_the_language(self, tree):
        by_name = {row["name"]: row["language"] for row in rsc.list_scripts(tree)}
        assert by_name["es/phonetics.txt"] == "es"
        assert by_name["en/general.txt"] == "en"

    def test_a_file_outside_a_language_directory_is_ignored(self, tree):
        """There is nowhere to file it, so listing it would offer an
        unrecordable script."""
        (tree / "loose.txt").write_text("orphan", encoding="utf8")
        assert all(row["name"] != "loose.txt" for row in rsc.list_scripts(tree))

    def test_an_unsupported_language_directory_is_ignored(self, tree):
        folder = tree / "fr"
        folder.mkdir()
        (folder / "bonjour.txt").write_text("Bonjour.", encoding="utf8")
        assert all(not row["name"].startswith("fr/") for row in rsc.list_scripts(tree))

    def test_resolves_a_nested_name(self, tree):
        assert rsc.resolve_script(tree, "es/phonetics.txt").name == "phonetics.txt"

    def test_rejects_a_traversal_through_a_language_directory(self, tree):
        with pytest.raises(wp.PipelineError):
            rsc.resolve_script(tree, "en/../../etc/passwd")

    def test_rejects_a_name_deeper_than_one_directory(self, tree):
        nested = tree / "en" / "deep"
        nested.mkdir()
        (nested / "x.txt").write_text("x", encoding="utf8")
        with pytest.raises(wp.PipelineError):
            rsc.resolve_script(tree, "en/deep/x.txt")

    def test_rejects_an_unsupported_language_directory(self, tree):
        folder = tree / "fr"
        folder.mkdir()
        (folder / "bonjour.txt").write_text("Bonjour.", encoding="utf8")
        with pytest.raises(wp.PipelineError):
            rsc.resolve_script(tree, "fr/bonjour.txt")


class TestScriptsDoNotShareTakes:
    """A take belongs to the script it was read from, not to its language.

    Regression: a clip was named for (language, index) alone, so recording
    line 5 of en/general.txt wrote en_00005.wav and every other en script
    answered 'yes, line 5 is recorded' by finding that file on disk. The
    picker showed a script nobody had opened as far along as the one being
    read.
    """

    def test_a_take_in_one_script_leaves_a_sibling_script_empty(
        self, scripts_dir, tmp_path
    ):
        text = script_of(4)
        directory = scripts_dir({"en/general.txt": text, "en/software.txt": text})
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        record(csv_path, audio_dir, "en/general.txt", "en", 0, wp.chunk_text(text)[0])

        other = rsc.script_progress(
            directory / "en" / "software.txt", csv_path, audio_dir, "en",
            "en/software.txt",
        )
        assert (other["recorded"], other["recorded_count"]) == (set(), 0)

    def test_the_script_that_was_read_still_reports_its_take(
        self, scripts_dir, tmp_path
    ):
        text = script_of(4)
        directory = scripts_dir({"en/general.txt": text, "en/software.txt": text})
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        record(csv_path, audio_dir, "en/general.txt", "en", 0, wp.chunk_text(text)[0])

        mine = rsc.script_progress(
            directory / "en" / "general.txt", csv_path, audio_dir, "en",
            "en/general.txt",
        )
        assert mine["recorded"] == {0}

    def test_two_scripts_record_the_same_index_without_overwriting(
        self, scripts_dir, tmp_path
    ):
        """Distinct files, or the second take would replace the first."""
        text = script_of(4)
        scripts_dir({"en/general.txt": text, "en/software.txt": text})
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        first = rs.clip_path(audio_dir, "en", 0, "en/general.txt")
        second = rs.clip_path(audio_dir, "en", 0, "en/software.txt")
        assert first != second

    def test_a_clip_name_survives_a_script_name_needing_escaping(self, tmp_path):
        """Script names carry a separator and a suffix; the file name may not."""
        clip = rs.clip_path(tmp_path, "en", 3, "en/soft ware.v2.txt")
        assert "/" not in clip.name and clip.parent == tmp_path


def record(csv_path, audio_dir, script, language, index, text):
    """Commit a take exactly as either front end does, without a microphone."""
    destination = rs.clip_path(audio_dir, language, index, script)
    destination.write_bytes(b"RIFF")
    rs.upsert_row(csv_path, destination, text, language)


class TestLegacyTakesKeepCounting:
    """Clips recorded before clips carried their script must not be orphaned.

    The user's library is named {language}_{index}.wav with no script in it,
    and prune_missing deletes any row whose file is gone - so a scheme that
    simply stopped looking for those names would have deleted every existing
    take on the next startup. The dataset's own text is the only surviving
    evidence of which script a legacy clip came from, so that is what decides.
    """

    def test_a_legacy_clip_counts_for_the_script_whose_text_it_matches(
        self, scripts_dir, tmp_path
    ):
        text = script_of(4)
        directory = scripts_dir({"en/general.txt": text})
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()
        legacy_take(csv_path, audio_dir, "en", 0, wp.chunk_text(text)[0])

        progress = rsc.script_progress(
            directory / "en" / "general.txt", csv_path, audio_dir, "en",
            "en/general.txt",
        )
        assert progress["recorded"] == {0}

    def test_a_legacy_clip_does_not_count_for_a_script_it_never_came_from(
        self, scripts_dir, tmp_path
    ):
        recorded_text = script_of(4)
        other_text = "\n\n".join(
            f"a completely different long sentence written for another script {n}."
            for n in range(4)
        )
        directory = scripts_dir(
            {"en/general.txt": recorded_text, "en/software.txt": other_text}
        )
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()
        legacy_take(csv_path, audio_dir, "en", 0, wp.chunk_text(recorded_text)[0])

        progress = rsc.script_progress(
            directory / "en" / "software.txt", csv_path, audio_dir, "en",
            "en/software.txt",
        )
        assert progress["recorded"] == set()

    def test_a_legacy_take_survives_a_prune(self, scripts_dir, tmp_path):
        """prune_missing must not read a legacy row as a clip gone missing."""
        text = script_of(4)
        scripts_dir({"en/general.txt": text})
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()
        legacy_take(csv_path, audio_dir, "en", 0, wp.chunk_text(text)[0])

        assert rs.prune_missing(csv_path, audio_dir) == 0

    def test_re_recording_replaces_the_legacy_file_rather_than_shadowing_it(
        self, scripts_dir, tmp_path
    ):
        """Two files for one line would leave the picker choosing between takes."""
        text = script_of(4)
        scripts_dir({"en/general.txt": text})
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()
        legacy_take(csv_path, audio_dir, "en", 0, wp.chunk_text(text)[0])

        target = rs.existing_clip_path(
            csv_path, audio_dir, "en", 0, "en/general.txt",
            dict(enumerate(wp.chunk_text(text))),
        )
        assert target == rs.clip_path(audio_dir, "en", 0)

    def test_a_fresh_line_is_written_under_the_scoped_name(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        audio_dir = tmp_path / "data"
        audio_dir.mkdir()

        target = rs.existing_clip_path(
            csv_path, audio_dir, "en", 0, "en/general.txt", {0: "anything"}
        )
        assert target == rs.clip_path(audio_dir, "en", 0, "en/general.txt")


def legacy_take(csv_path, audio_dir, language, index, text):
    """A take named the way the recorder named them before scripts scoped clips."""
    destination = rs.clip_path(audio_dir, language, index)
    destination.write_bytes(b"RIFF")
    rs.upsert_row(csv_path, destination, text, language)
