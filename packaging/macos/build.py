"""Build the local macOS prototype with the current Python environment."""

import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 11):
        raise SystemExit("Build on macOS with Python 3.11. Cross-compilation is not supported.")
    root = Path(__file__).resolve().parents[2]
    work = root / "build" / "macos"
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(work / "pyinstaller-cache")
    env["MPLCONFIGDIR"] = str(work / "matplotlib")
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(root / "dist" / "macos"), "--workpath", str(work),
        str(root / "packaging" / "macos" / "Arte-Profiler.spec"),
    ], cwd=root, env=env, check=True)
    versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (root / "dist" / "macos" / "build-environment.txt").write_text(
        f"Python {sys.version}\nArchitecture: {platform.machine()}\n{versions}", encoding="utf-8",
    )
    print(f"Built {root / 'dist' / 'macos' / 'Arte-Profiler.app'}")


if __name__ == "__main__":
    main()
