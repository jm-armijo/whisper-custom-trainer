"""Checks on the container that runs the web recorder on the DietPi box.

Most assertions are static: the image contents are decided by text in the
Dockerfile and compose file, and reading that text costs nothing. The two tests
that need Docker itself skip with an explicit reason rather than passing
vacuously - a `which`-guarded test that silently succeeds when the tool is
missing hid a real failure in this repo before (see CLAUDE.md on
ct2-transformers-converter).
"""

import shutil
import subprocess

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration

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
        for flag in ("--scripts-dir", "--out-dir", "--csv", "--host", "--port"):
            assert flag in dockerfile


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
