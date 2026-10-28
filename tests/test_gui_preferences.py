"""Persistent browse locations are separate from scientific inputs and outputs."""

import json
from pathlib import Path

from arte_profiler.gui import preferences


def test_history_survives_reads_and_keeps_inputs_and_results_separate(tmp_path, monkeypatch):
    path = tmp_path / "settings" / "browse.json"
    monkeypatch.setattr(preferences, "_preferences_path", lambda: path)
    preferences.remember_directory(tmp_path / "inputs", False)
    preferences.remember_directory(tmp_path / "results", True)
    assert preferences.last_directory(False) == str(tmp_path / "inputs")
    assert preferences.last_directory(True) == str(tmp_path / "results")
    assert json.loads(path.read_text()) == {
        "inputs": str(tmp_path / "inputs"), "results": str(tmp_path / "results"),
    }
    assert list(path.parent.iterdir()) == [path]


def test_damaged_history_is_ignored_and_can_be_replaced(tmp_path, monkeypatch):
    path = tmp_path / "browse.json"
    monkeypatch.setattr(preferences, "_preferences_path", lambda: path)
    for content in ("{", "[]", '{"inputs": 123}'):
        path.write_text(content)
        assert preferences.last_directory(False) is None
    preferences.remember_directory(tmp_path, False)
    assert preferences.last_directory(False) == str(tmp_path)


def test_failed_history_write_does_not_prevent_browsing(tmp_path, monkeypatch, caplog):
    parent = tmp_path / "unwritable"
    monkeypatch.setattr(preferences, "_preferences_path", lambda: parent / "browse.json")
    original = Path.mkdir

    def denied(self, *args, **kwargs):
        if self == parent:
            raise PermissionError("Denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    preferences.remember_directory(tmp_path, False)
    assert "Could not save the last browse folder" in caplog.text
