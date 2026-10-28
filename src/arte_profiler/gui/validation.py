"""Read-only GUI preflight; never decode pixels or change scientific inputs."""

from pathlib import Path
import stat

import tifffile


def file_error(value: str, label: str, suffixes: tuple[str, ...], *, optional: bool = False,
               tiff: bool = False) -> str:
    allowed = " or ".join(suffixes)
    if not value.strip():
        return "" if optional else f"Select the {label[0].lower() + label[1:]} ({allowed})."
    try:
        path = Path(value).expanduser()
        if path.suffix.lower() not in suffixes:
            extension = path.suffix or "no extension"
            return f"{label}: unsupported file extension ({extension}). Select a {allowed} file."
        if not stat.S_ISREG(path.stat().st_mode):
            return f"{label}: the selected path is not a file: {path}"
        with path.open("rb") as stream:
            if tiff:
                # Read directory/tag metadata only. Pixel decoding stays in the backend.
                with tifffile.TiffFile(stream) as image:
                    if not image.pages or not image.pages[0].imagewidth or not image.pages[0].imagelength:
                        return f"{label}: invalid TIFF file; no readable image page was found."
        return ""
    except FileNotFoundError:
        return f"{label}: file not found: {value}"
    except PermissionError:
        return f"{label}: cannot read this file. Check access permissions: {value}"
    except (OSError, ValueError, IndexError) as error:
        kind = "invalid or unreadable TIFF file" if tiff else "cannot open this file"
        return (f"{label}: {kind}. Check that the file is available locally or on the network. "
                f"Details: {error}")
