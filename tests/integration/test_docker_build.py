"""Checks on the container that runs the web recorder on the DietPi box.

Most assertions are static: the image contents are decided by text in the
Dockerfile and compose file, and reading that text costs nothing. The two tests
that need Docker itself skip with an explicit reason rather than passing
vacuously - a `which`-guarded test that silently succeeds when the tool is
missing hid a real failure in this repo before (see CLAUDE.md on
ct2-transformers-converter).
"""

import ast
import re
import shutil
import subprocess

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration

SERVER_MODULE = "recorder_server"

DOCKERFILE = wp.PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = wp.PROJECT_ROOT / "docker-compose.yml"
DOCKERIGNORE = wp.PROJECT_ROOT / ".dockerignore"

# Installing any of these would pull the training stack (~2GB) into an image
# that only records, and torch has no wheel for some 32-bit ARM DietPi targets.
TRAINING_ONLY_PACKAGES = ("torch", "transformers", "peft", "datasets",
                          "accelerate", "ctranslate2", "faster-whisper")


def docker_cli():
    """The docker binary, or a skip naming what is missing."""
    binary = shutil.which("docker")
    if binary is None:
        pytest.skip("docker is not installed; cannot exercise the real CLI")
    return binary


def docker_daemon():
    """A docker binary talking to a live daemon, or a skip naming which is absent."""
    binary = docker_cli()
    probe = subprocess.run([binary, "info"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        pytest.skip(f"docker daemon unreachable: {probe.stderr.strip().splitlines()[-1:]}")
    return binary


def local_imports_of(module, seen=None):
    """Every project module reachable from `module` by import, including itself.

    Parsed rather than imported: importing the server would bind its socket.
    """
    seen = set() if seen is None else seen
    path = wp.PROJECT_ROOT / f"{module}.py"
    if module in seen or not path.exists():
        return seen
    seen.add(module)

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        for name in names:
            if (wp.PROJECT_ROOT / f"{name}.py").exists():
                local_imports_of(name, seen)
    return seen


def server_modules():
    """The server's local import graph, or a skip until that module lands.

    Only the two tests that read the server's imports depend on it; the
    packaging assertions hold whether or not it exists yet, so this is a
    per-test guard rather than a module-wide one.
    """
    if not (wp.PROJECT_ROOT / f"{SERVER_MODULE}.py").exists():
        pytest.skip(f"{SERVER_MODULE}.py does not exist yet; cannot read its imports")
    return local_imports_of(SERVER_MODULE)


def instructions(source):
    """The Dockerfile's build instructions, with comments and blanks dropped.

    What lands in the image is decided by the instructions alone. Matching raw
    text would let a comment explaining why torch is *absent* fail the test
    asserting torch is absent.
    """
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def dockerfile():
    return instructions(DOCKERFILE.read_text())


@pytest.fixture(scope="module")
def compose():
    return COMPOSE_FILE.read_text()


class TestImageStaysSmall:
    """The container records; training happens on the laptop."""

    @pytest.mark.parametrize("package", TRAINING_ONLY_PACKAGES)
    def test_omits_the_training_stack(self, dockerfile, package):
        assert package not in dockerfile

    def test_installs_what_decoding_an_upload_needs(self, dockerfile):
        assert all(package in dockerfile for package in ("numpy", "librosa", "soundfile"))

    def test_installs_ffmpeg_for_the_browsers_webm_opus(self, dockerfile):
        """MediaRecorder emits WebM/Opus, which libsndfile alone cannot decode."""
        assert "ffmpeg" in dockerfile

    def test_omits_sounddevice(self, dockerfile):
        """The browser captures the mic; the container has no audio device."""
        assert "sounddevice" not in dockerfile

    def test_copies_only_the_modules_the_server_imports(self, dockerfile):
        """train/merge/export import libraries this image does not install."""
        copied = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
        assert copied, "the image must copy the application in"
        assert not any(module in line for line in copied
                       for module in ("train.py", "merge.py", "export.py", "record_data.py"))

    def test_copies_every_local_module_the_server_needs(self, dockerfile):
        """A module left out crashes the container on import, not at build time."""
        copied = "\n".join(line for line in dockerfile.splitlines() if line.startswith("COPY "))
        for module in server_modules():
            assert f"{module}.py" in copied, f"{module} is imported but never COPYed"

    def test_the_copied_modules_import_nothing_uninstalled(self, dockerfile):
        """Heavy imports live inside functions, so module scope must stay clean.

        librosa is installed; torch and transformers are not, and a top-level
        import of either would break the container the first time it started.
        """
        for module in server_modules():
            source = (wp.PROJECT_ROOT / f"{module}.py").read_text()
            top_level = [line for line in source.splitlines()
                         if line.startswith(("import ", "from "))]
            for package in TRAINING_ONLY_PACKAGES:
                assert not any(line.startswith((f"import {package}", f"from {package}"))
                               for line in top_level), f"{module} imports {package} at module scope"

    def test_the_start_command_only_passes_flags_the_server_accepts(self, dockerfile):
        """A flag the parser does not define kills the container on startup.

        The Dockerfile and recorder_server.py were written separately, and the
        CMD shipped `--scripts-dir` against a parser defining `--scripts`:
        `docker compose up` exited immediately on 'unrecognized arguments'
        while the image built and pushed perfectly happily.
        """
        if not (wp.PROJECT_ROOT / f"{SERVER_MODULE}.py").exists():
            pytest.skip(f"{SERVER_MODULE}.py does not exist yet; cannot read its flags")

        # The CMD spans several backslash-continued lines, so continuations are
        # folded first; scanning line by line would see only `exec python ...`.
        folded = re.sub(r"\\\s*\n\s*", " ", dockerfile)
        command = "\n".join(line for line in folded.splitlines()
                            if line.startswith(("CMD", "ENTRYPOINT")))
        passed = set(re.findall(r"--[a-z][a-z-]*", command))
        assert passed, "the image must start the server with explicit flags"

        source = (wp.PROJECT_ROOT / f"{SERVER_MODULE}.py").read_text()
        defined = set(re.findall(r"add_argument\(\s*[\"'](--[a-z][a-z-]*)[\"']", source))
        assert defined, "could not read the server's argument parser"

        unknown = passed - defined
        assert not unknown, f"CMD passes flags the server rejects: {sorted(unknown)}"

    def test_excludes_the_heavy_directories_from_the_build_context(self):
        """The context is uploaded before the first layer; venv/ and models are GBs."""
        ignored = DOCKERIGNORE.read_text()
        assert all(entry in ignored for entry in
                   ("venv/", "merged-whisper-model/", "exports/", "data/", ".git/"))


class TestMultiArch:
    def test_uses_a_multi_arch_base(self, dockerfile):
        """DietPi is usually arm64; an amd64-only base would not build there."""
        bases = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert bases == ["FROM python:3.12-slim"]

    def test_does_not_pin_a_single_platform(self, dockerfile):
        """--platform on FROM would defeat the multi-arch manifest."""
        assert "--platform" not in dockerfile


class TestNetworking:
    def test_binds_every_interface_inside_the_container(self, dockerfile):
        """Bound to 127.0.0.1 the published port would answer nothing."""
        assert "0.0.0.0" in dockerfile

    def test_publishes_the_port(self, compose):
        assert ":8080\"" in compose

    def test_declares_the_port(self, dockerfile):
        assert "EXPOSE 8080" in dockerfile


class TestPersistence:
    """A rebuild or `compose down` destroys the writable layer; takes must not go with it."""

    def test_mounts_the_audio_directory(self, compose):
        assert "./data:/data/audio" in compose

    def test_mounts_the_dataset_csv(self, compose):
        assert "./dataset.csv:/data/dataset.csv" in compose

    def test_mounts_the_reading_material_read_only(self, compose):
        """The server only reads scripts/; :ro makes an accidental write fail loudly."""
        assert "./scripts:/data/scripts:ro" in compose

    def test_passes_every_path_to_the_server(self, dockerfile):
        """Each mount must reach the server, addressed by the env var it reads.

        Asserted through the RECORDER_* variables rather than the flag names:
        hardcoding a spelling here is what let `--scripts-dir` pass this test
        while the server's parser only ever defined `--scripts`. The companion
        test above checks the spellings against the parser itself.
        """
        for variable in ("RECORDER_SCRIPTS_DIR", "RECORDER_OUT_DIR", "RECORDER_CSV",
                         "RECORDER_HOST", "RECORDER_PORT"):
            assert variable in dockerfile


class TestComposeIsValid:
    def test_compose_file_parses(self):
        """`docker compose config` is the only authority on this schema."""
        binary = docker_cli()
        result = subprocess.run(
            [binary, "compose", "-f", str(COMPOSE_FILE), "config"],
            capture_output=True, text=True, cwd=wp.PROJECT_ROOT, check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_image_builds(self):
        """Kept out of the fast tier: a cold build fetches a base image."""
        binary = docker_daemon()
        result = subprocess.run(
            [binary, "build", "-f", str(DOCKERFILE), "-t", "my-whisper-recorder:test", "."],
            capture_output=True, text=True, cwd=wp.PROJECT_ROOT, check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
