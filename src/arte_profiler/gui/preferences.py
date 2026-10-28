"""Small per-user browser preferences, independent of scientific settings."""

import json
from contextlib import suppress
import logging
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock

_lock = Lock()


def _preferences_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Arte-Profiler"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Arte-Profiler"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "arte-profiler"
    return base / "browse.json"


def _read() -> dict[str, str]:
    try:
        data = json.loads(_preferences_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {key: value for key, value in data.items()
                    if key in {"inputs", "results"} and isinstance(value, str) and value}
    except (OSError, ValueError):
        pass
    return {}


def last_directory(directory: bool) -> str | None:
    with _lock:
        return _read().get("results" if directory else "inputs")


def remember_directory(path: Path, directory: bool) -> None:
    """Replace atomically; a failed preference write must not prevent browsing."""
    temporary = None
    with _lock:
        try:
            preferences = _preferences_path()
            data = _read()
            data["results" if directory else "inputs"] = str(path)
            preferences.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=preferences.parent,
                                             prefix="browse-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False)
            temporary.replace(preferences)
        except OSError:
            logging.getLogger(__name__).warning("Could not save the last browse folder", exc_info=True)
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
