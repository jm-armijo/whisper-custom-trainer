"""Regression test: a converter that exits 0 without output fails loudly."""

from types import SimpleNamespace

import pytest

import export
import whisper_pipeline as wp


class TestGgmlExport:
    """The rename step must not surface a bare FileNotFoundError."""

    def test_reports_a_converter_that_wrote_nothing(self, tmp_path, monkeypatch):
        converter = tmp_path / "convert-h5-to-ggml.py"
        converter.write_text("")
        monkeypatch.setattr(wp, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(wp, "MERGED_MODEL_DIR", tmp_path / "merged")
        (tmp_path / "exports").mkdir()
        # A converter that "succeeds" while producing no file.
        monkeypatch.setattr(
            export.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
        )

        with pytest.raises(wp.PipelineError, match="wrote no"):
            export.export_ggml()
