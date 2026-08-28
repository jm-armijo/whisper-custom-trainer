"""Static checks on the shell entry points; running them would mutate the machine."""

import subprocess

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration

SCRIPTS = ("setup.sh", "convert.sh")


@pytest.mark.parametrize("script", SCRIPTS)
class TestShellSyntax:
    def test_parses_without_syntax_errors(self, script):
        subprocess.run(["bash", "-n", str(wp.PROJECT_ROOT / script)], check=True)

    def test_is_executable(self, script):
        assert (wp.PROJECT_ROOT / script).stat().st_mode & 0o111

    def test_aborts_on_error(self, script):
        """Without -e a failed step would be reported as success."""
        assert "set -euo pipefail" in (wp.PROJECT_ROOT / script).read_text()


class TestSetupScript:
    @pytest.fixture(scope="class")
    @classmethod
    def source(cls):
        return (wp.PROJECT_ROOT / "setup.sh").read_text()

    def test_installs_every_runtime_dependency(self, source):
        required = ("torch", "transformers", "peft", "datasets", "accelerate",
                    "librosa", "soundfile", "sounddevice")
        assert all(package in source for package in required)

    def test_installs_the_export_toolchain(self, source):
        assert "ctranslate2" in source and "faster-whisper" in source

    def test_clones_from_the_current_whisper_cpp_org(self, source):
        """ggerganov/whisper.cpp redirects; ggml-org is canonical."""
        assert "ggml-org/whisper.cpp" in source

    def test_removes_the_superseded_environment(self, source):
        assert "whisper-env" in source


class TestConvertScript:
    @pytest.fixture(scope="class")
    @classmethod
    def source(cls):
        return (wp.PROJECT_ROOT / "convert.sh").read_text()

    def test_installs_the_correct_cask_name(self, source):
        """The package is openwhispr; open-wispr does not exist."""
        assert "--cask openwhispr" in source

    def test_installs_into_the_directory_openwhispr_reads(self, source):
        assert ".cache/openwhispr/whisper-models" in source

    def test_uses_a_registry_accepted_filename(self, source):
        """OpenWhispr rejects any model name outside its fixed registry."""
        assert "ggml-small.bin" in source

    def test_backs_up_the_stock_model(self, source):
        assert ".orig" in source
