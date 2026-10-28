"""Packaging configuration, runtime isolation and opt-in frozen resource smoke test."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_packaging_includes_only_required_scientific_resources():
    spec = importlib.util.spec_from_file_location("macos_resources", ROOT / "packaging" / "macos" / "resources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datas, binaries = module.package_resources(ROOT)
    assert {Path(source).name for source, _ in binaries} == {"scanin", "colprof", "icclu"}
    assert all("/macos/bin/" in source for source, _ in binaries)
    assert any(source.endswith("/data/guidelines/guidelines.yaml") for source, _ in datas)
    assert sum(source.endswith(".tiff") for source, _ in datas) == 3
    assert sum(source.endswith(".cie") for source, _ in datas) == 3
    assert sum(source.endswith(".ttf") for source, _ in datas) == 2
    assert not any("targets_old" in source or "/profiles/" in source for source, _ in datas)
    for source, destination in [*datas, *binaries]:
        assert Path(source).is_file()
        assert not Path(destination).is_absolute()


def test_runtime_writes_to_user_cache_and_logs(tmp_path, monkeypatch):
    from arte_profiler.gui import packaged

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in ("MPLBACKEND", "MPLCONFIGDIR", "NICEGUI_STORAGE_PATH"):
        monkeypatch.setenv(name, "previous-value")
    packaged.prepare_runtime()
    cache, logs = packaged.runtime_paths()
    assert cache.is_dir() and logs.is_dir()
    assert Path.cwd() == cache
    assert os.environ["MPLBACKEND"] == "Agg"
    assert Path(os.environ["MPLCONFIGDIR"]).is_relative_to(cache)
    assert Path(os.environ["NICEGUI_STORAGE_PATH"]).is_relative_to(cache)


def test_bundle_output_guard_resolves_symlinks(tmp_path, monkeypatch):
    from arte_profiler.gui import packaged

    bundle = tmp_path / "Arte-Profiler.app"
    resources = bundle / "Contents" / "Resources"
    resources.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "Contents" / "MacOS" / "Arte-Profiler"))
    link = tmp_path / "linked-resources"
    link.symlink_to(resources, target_is_directory=True)
    for path in (resources, link / "results"):
        with pytest.raises(ValueError, match="outside the Arte-Profiler application bundle"):
            packaged.protect_bundle_output(path)
    packaged.protect_bundle_output(tmp_path / "results")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS launcher lock")
def test_second_launch_reopens_existing_instance_and_lock_is_released(tmp_path, monkeypatch):
    from arte_profiler.gui import packaged

    opened = []
    monkeypatch.setattr(packaged.webbrowser, "open", opened.append)
    session = tmp_path / "session.json"
    with packaged.single_instance(tmp_path, open_browser=True) as acquired:
        assert acquired
        session.write_text(json.dumps({"url": "http://127.0.0.1:12345/"}))
        with packaged.single_instance(tmp_path, open_browser=True) as second:
            assert not second
        assert opened == ["http://127.0.0.1:12345/"]
    assert not session.exists()
    with packaged.single_instance(tmp_path, open_browser=False) as acquired:
        assert acquired


@pytest.mark.skipif(not os.environ.get("ARTE_PROFILER_PACKAGED_APP"), reason="Opt-in built macOS application")
def test_frozen_resources_tools_and_spawned_worker(tmp_path):
    bundle = Path(os.environ["ARTE_PROFILER_PACKAGED_APP"]).resolve()
    executable = bundle / "Contents" / "MacOS" / "Arte-Profiler"
    # No project/Python/Conda environment on the executable's search path.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "CONDA", "DYLD", "MPL", "NICEGUI"))}
    env.update(HOME=str(tmp_path), PATH="/usr/bin:/bin", TMPDIR=str(tmp_path))
    output = tmp_path / "smoke"
    completed = subprocess.run([str(executable), "--smoke-test", str(output)], cwd=tmp_path,
                               env=env, capture_output=True, text=True, timeout=180)
    logs = tmp_path / "Library" / "Logs" / "Arte-Profiler" / "startup.log"
    assert completed.returncode == 0, logs.read_text() if logs.exists() else completed.stderr
    result = json.loads((output / "smoke.json").read_text())
    for process in result.values():
        assert process["frozen"]
        assert Path(process["package"]).is_relative_to(bundle)
        assert set(process["argyll"]) == {"scanin", "colprof", "icclu"}
        assert process["targets"] == ["ColorCheckerSG", "ColorCheckerPassport", "DT-NGT2"]
    assert result["parent"]["pid"] != result["worker"]["pid"]
