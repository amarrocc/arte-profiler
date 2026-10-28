"""macOS bundle lifecycle and diagnostics; scientific APIs remain unchanged."""

import argparse
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import webbrowser


def runtime_paths() -> tuple[Path, Path]:
    home = Path.home()
    return home / "Library" / "Caches" / "Arte-Profiler", home / "Library" / "Logs" / "Arte-Profiler"


def prepare_runtime() -> None:
    """Keep caches, windowless console output and incidental writes outside the bundle."""
    cache, logs = runtime_paths()
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["NICEGUI_STORAGE_PATH"] = str(cache / "nicegui")
    os.chdir(cache)
    if getattr(sys, "frozen", False) or sys.stdout is None:
        sys.stdout = (logs / "startup.log").open("a", encoding="utf-8", buffering=1)
    if getattr(sys, "frozen", False) or sys.stderr is None:
        sys.stderr = (logs / "startup.log").open("a", encoding="utf-8", buffering=1)


@contextmanager
def single_instance(cache: Path, *, open_browser: bool):
    """Hold an OS lock; a second launch reopens the running instance."""
    import fcntl

    with (cache / "instance.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            session = cache / "session.json"
            if open_browser and session.exists():
                webbrowser.open(json.loads(session.read_text())["url"])
            yield False
            return
        session = cache / "session.json"
        session.unlink(missing_ok=True)
        try:
            yield True
        finally:
            session.unlink(missing_ok=True)


def resource_probe() -> dict:
    """Exercise the same resource resolver and executables used by the scientific API."""
    from importlib.resources import files

    import imagecodecs
    from nicegui import ui
    import yaml
    from arte_profiler import profiling, profiling_utils

    package = Path(str(files("arte_profiler")))
    targets = package / "data" / "targets"
    manifest = yaml.safe_load((targets / "targets_manifest.yaml").read_text())
    required = [package / "data" / "guidelines" / "guidelines.yaml"]
    for target in manifest.values():
        required.extend(targets / target[key] for key in ("image_path", "chart_cht", "chart_cie"))
    required.extend(package / "tools" / "dejavu-sans_font" / name for name in (
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    ))
    nicegui = Path(str(files("nicegui")))
    required.extend([nicegui / "templates" / "index.html", nicegui / "static" / "quasar.umd.prod.js"])
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty packaged resource: {path}")
    versions = {}
    for name in ("scanin", "colprof", "icclu"):
        path = Path(profiling_utils.get_argyll_bin_path()) / name
        if not os.access(path, os.X_OK):
            raise PermissionError(f"ArgyllCMS tool is not executable: {path}")
        try:
            result = subprocess.run([str(path), "-?"], capture_output=True, text=True, timeout=20)
        except OSError as error:
            raise RuntimeError(f"Cannot run {name}; Intel ArgyllCMS requires Rosetta on Apple Silicon: {error}") from error
        text = result.stdout + result.stderr
        if result.returncode not in (0, 1) or "usage:" not in text.lower():
            raise RuntimeError(f"ArgyllCMS {name} failed to start: {text}")
        versions[name] = text.splitlines()[0]
    # Importing profiling above also registers the two report fonts.
    return {
        "frozen": bool(getattr(sys, "frozen", False)), "pid": os.getpid(),
        "package": str(package), "targets": list(manifest), "argyll": versions,
        "resources_checked": len(required), "imagecodecs": imagecodecs.__version__,
    }


def _probe_worker(report: Path) -> None:
    try:
        result = resource_probe()
    except Exception:
        report.write_text(json.dumps({"error": traceback.format_exc()}, indent=2))
        raise
    report.write_text(json.dumps(result, indent=2))


def smoke_test(directory: Path) -> None:
    """A frozen subprocess must import science/resources without launching another server."""
    directory.mkdir(parents=True, exist_ok=True)
    parent = resource_probe()
    report = directory / "worker.json"
    worker = multiprocessing.get_context("spawn").Process(target=_probe_worker, args=(report,))
    worker.start()
    worker.join(timeout=90)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise RuntimeError("Packaged worker did not finish within 90 seconds")
    if worker.exitcode != 0:
        raise RuntimeError(f"Packaged worker failed: {report.read_text() if report.exists() else worker.exitcode}")
    child = json.loads(report.read_text())
    if parent["pid"] == child["pid"]:
        raise RuntimeError("Smoke test did not spawn a separate worker")
    (directory / "smoke.json").write_text(json.dumps({"parent": parent, "worker": child}, indent=2))
    worker.close()


def protect_bundle_output(directory: Path) -> None:
    """Reject a results folder inside the frozen application, including symlinks."""
    if getattr(sys, "frozen", False):
        bundle = Path(sys.executable).resolve().parents[2]
        if directory.resolve().is_relative_to(bundle):
            raise ValueError("Select a results folder outside the Arte-Profiler application bundle.")


def request_quit() -> None:
    from nicegui import app, ui

    def quit_if_idle() -> None:
        if multiprocessing.active_children():
            ui.notify("Processing is running. Wait for it to finish before quitting.", type="warning")
            return
        dialog.close()
        app.shutdown()

    if multiprocessing.active_children():
        ui.notify("Processing is running. Wait for it to finish before quitting.", type="warning")
        return
    with ui.dialog() as dialog, ui.card().classes("gap-3"):
        ui.label("Quit Arte-Profiler?").classes("text-lg font-medium")
        ui.label("This stops the local application. You can then close the browser tab.")
        with ui.row():
            ui.button("Keep open", on_click=dialog.close).props("flat")
            ui.button("Quit", on_click=quit_if_idle).mark("confirm-quit")
    dialog.on_value_change(lambda event: dialog.delete() if not event.value else None)
    dialog.open()


def packaged_page() -> None:
    from nicegui import ui
    from .app import create_page

    with ui.row().classes("w-full max-w-4xl mx-auto justify-end px-4 sm:px-8"):
        ui.button("Quit Arte-Profiler", icon="power_settings_new", on_click=request_quit).props("flat")
    create_page()


def main() -> None:
    parser = argparse.ArgumentParser(description="Arte-Profiler experimental macOS application")
    parser.add_argument("--smoke-test", type=Path, metavar="DIRECTORY", help="Check bundled resources and a spawned worker; then exit")
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser (diagnostics)")
    args = parser.parse_args()
    try:
        if args.smoke_test:
            smoke_test(args.smoke_test.absolute())
            return
        cache, _ = runtime_paths()
        with single_instance(cache, open_browser=not args.no_browser) as acquired:
            if not acquired:
                return
            resource_probe()
            from nicegui import ui

            ui.page("/")(packaged_page)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            (cache / "session.json").write_text(json.dumps({"pid": os.getpid(), "url": f"http://127.0.0.1:{port}/"}))
            ui.run(host="127.0.0.1", port=port, title="Arte-Profiler", show=not args.no_browser,
                   native=False, reload=False)
    except Exception:
        _, logs = runtime_paths()
        with (logs / "startup.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        if not args.smoke_test and not args.no_browser:
            subprocess.run([
                "/usr/bin/osascript", "-e",
                'on run argv\ndisplay alert "Arte-Profiler could not start" message (item 1 of argv) as critical\nend run',
                f"See {logs / 'startup.log'} for details.",
            ], check=False)
        raise
