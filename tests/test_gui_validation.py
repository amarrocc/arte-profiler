"""GUI preflight checks actual files without decoding pixels or running science."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from arte_profiler.gui.validation import file_error


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_tiff_preflight_reads_metadata_only(tmp_path, monkeypatch, dtype):
    path = tmp_path / "capture.TIFF"
    tifffile.imwrite(path, np.zeros((8, 8, 3), dtype=dtype), photometric="rgb")

    def no_pixels(*args, **kwargs):
        pytest.fail("GUI preflight must not decode image pixels")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", no_pixels)
    assert file_error(str(path), "Profiling TIFF image", (".tif", ".tiff"), tiff=True) == ""


@pytest.mark.parametrize("value,expected", [
    ("", "Select the profiling TIFF image"),
    ("   ", "Select the profiling TIFF image"),
    ("capture.jpg", "unsupported file extension (.jpg)"),
    ("capture", "unsupported file extension (no extension)"),
    ("missing.tiff", "file not found"),
])
def test_distinct_file_selection_errors(tmp_path, monkeypatch, value, expected):
    monkeypatch.chdir(tmp_path)
    assert expected in file_error(value, "Profiling TIFF image", (".tif", ".tiff"), tiff=True)


@pytest.mark.parametrize("content", [b"", b"not a TIFF", b"\xff\xd8\xff\xe0JPEG", b"II*\x00\x00\x00\x00\x00"])
def test_invalid_empty_or_disguised_tiff(tmp_path, content):
    path = tmp_path / "broken.tiff"
    path.write_bytes(content)
    message = file_error(str(path), "Evaluation TIFF image", (".tif", ".tiff"), tiff=True)
    assert "TIFF" in message
    assert "invalid" in message


@pytest.mark.parametrize("suffix,label", [(".cie", "Profiling CIE reference"), (".icc", "Existing ICC profile")])
def test_reference_and_icc_existence_without_scientific_parsing(tmp_path, suffix, label):
    path = tmp_path / f"reference{suffix}"
    assert "file not found" in file_error(str(path), label, (suffix,))
    path.write_text("No scientific parsing in GUI preflight")
    assert file_error(str(path), label, (suffix,)) == ""
    assert file_error("", label, (suffix,), optional=True) == ""


def test_permission_error_is_distinct(tmp_path, monkeypatch):
    path = tmp_path / "capture.tiff"
    path.touch()
    original = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("Denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    assert "Check access permissions" in file_error(str(path), "Profiling TIFF image", (".tiff",), tiff=True)
