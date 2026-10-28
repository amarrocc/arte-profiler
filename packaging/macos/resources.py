"""Explicit scientific resources; exclude local experiments and other platforms."""

from pathlib import Path

import yaml


def package_resources(root: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    package = root / "src" / "arte_profiler"
    targets = package / "data" / "targets"
    manifest = targets / "targets_manifest.yaml"
    required = {manifest, package / "data" / "guidelines" / "guidelines.yaml"}
    for target in yaml.safe_load(manifest.read_text()).values():
        required.update(targets / target[key] for key in ("image_path", "chart_cht", "chart_cie"))
    required.update(package / "tools" / "dejavu-sans_font" / name for name in (
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVu Fonts License.txt",
    ))
    argyll = package / "tools" / "argyllcms_v3.3.0" / "macos"
    required.update(argyll / name for name in ("License.txt", "Readme.txt"))
    binaries = [argyll / "bin" / name for name in ("scanin", "colprof", "icclu")]
    for path in [*required, *binaries, root / "LICENSE"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required packaging resource is missing: {path}")
    for path in binaries:
        if not path.stat().st_mode & 0o111:
            raise PermissionError(f"ArgyllCMS executable is not executable: {path}")
    def destination(path: Path) -> tuple[str, str]:
        return str(path), str(Path("arte_profiler") / path.relative_to(package).parent)

    return [destination(path) for path in sorted(required)] + [(str(root / "LICENSE"), ".")], [
        destination(path) for path in binaries
    ]
