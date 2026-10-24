"""A local filesystem picker using NiceGUI's existing dialog and table elements."""

import os
from pathlib import Path

from nicegui import events, run, ui


def _entries(path: Path, suffixes: tuple[str, ...], directory: bool) -> list[dict[str, str]]:
    rows = []
    for entry in path.iterdir():
        try:
            is_directory = entry.is_dir()
            if not is_directory and (directory or entry.suffix.lower() not in suffixes):
                continue
            if is_directory or entry.is_file():
                rows.append({
                    "name": entry.name, "kind": "Folder" if is_directory else "File",
                    "path": str(entry),
                })
        except OSError:
            continue
    return sorted(rows, key=lambda row: (row["kind"] != "Folder", row["name"].casefold()))


def _initial_directory(value: str) -> Path:
    try:
        path = Path(os.path.abspath(os.path.expanduser(value))) if value else Path.home()
        while not path.is_dir() and path != path.parent:
            path = path.parent
        return path
    except (OSError, ValueError):
        return Path.home()


class _PathPicker(ui.dialog):
    def __init__(self, value: str, title: str, suffixes: tuple[str, ...], directory: bool):
        super().__init__()
        self.path = _initial_directory(value)
        self.suffixes = suffixes
        self.directory = directory
        self._navigation = 0
        with self, ui.card().classes("w-full gap-4").style("width: 760px; max-width: 95vw"):
            ui.label(f"Browse: {title}").classes("text-lg font-medium")
            with ui.row().classes("w-full flex-nowrap items-center"):
                self.location = ui.input("Location", value=str(self.path)).props("outlined").classes(
                    "flex-1 min-w-0"
                ).on("keydown.enter", self._go)
                ui.button("Go", on_click=self._go).props("outline")
            with ui.row():
                self.up = ui.button("Up", icon="arrow_upward", on_click=self._up).props("flat")
                ui.button("Home", icon="home", on_click=self._home).props("flat")
            ui.label(
                "Click a folder to open it." if directory
                else f"Click a folder to open it or a file to select it. Showing: {', '.join(suffixes)}."
            ).classes("text-sm text-slate-600")
            self.error = ui.label().classes("text-sm text-red-800").mark("picker-error")
            self.table = ui.table(
                columns=[
                    {"name": "name", "label": "Name", "field": "name", "align": "left"},
                    {"name": "kind", "label": "Type", "field": "kind", "align": "left"},
                ],
                rows=[], row_key="path", pagination=12,
            ).props("flat bordered dense").classes("w-full cursor-pointer").on("rowClick", self._choose)
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=self.close).props("outline")
                self.select_folder = ui.button("Select this folder", on_click=self._select_folder)
                self.select_folder.set_visibility(directory)
                self.select_folder.disable()

    async def navigate(self, path: Path) -> None:
        self._navigation += 1
        navigation = self._navigation
        self.table.props("loading")
        try:
            path = Path(os.path.abspath(path.expanduser()))
            rows = await run.io_bound(_entries, path, self.suffixes, self.directory)
        except (OSError, ValueError) as error:
            if navigation == self._navigation and not self.is_deleted:
                self.error.set_text(f"Cannot open folder: {error}")
                self.location.value = str(self.path)
        else:
            if navigation == self._navigation and not self.is_deleted:
                self.path = path
                self.location.value = str(path)
                self.table.rows = rows
                self.table.pagination = {"rowsPerPage": 12, "page": 1}
                self.error.set_text("")
                self.up.set_enabled(path != path.parent)
                self.select_folder.enable()
        finally:
            if navigation == self._navigation and not self.is_deleted:
                self.table.props(remove="loading")

    async def _go(self) -> None:
        if not self.location.value:
            self.error.set_text("Enter a folder path.")
            return
        path = Path(self.location.value).expanduser()
        await self.navigate(path if path.is_absolute() else self.path / path)

    async def _up(self) -> None:
        await self.navigate(self.path.parent)

    async def _home(self) -> None:
        await self.navigate(Path.home())

    async def _choose(self, event: events.GenericEventArguments) -> None:
        selected = event.args[1]
        row = next((row for row in self.table.rows if row["path"] == selected.get("path")), None)
        if row is None:
            return
        path = Path(row["path"])
        if row["kind"] == "Folder":
            await self.navigate(path)
        elif path.is_file():
            self.submit(str(path))
        else:
            self.error.set_text("That file is no longer available. Choose another file.")

    def _select_folder(self) -> None:
        if self.path.is_dir():
            self.submit(str(self.path))
        else:
            self.error.set_text("That folder is no longer available. Choose another folder.")


async def pick_path(
    value: str, *, title: str, suffixes: tuple[str, ...] = (), directory: bool = False,
) -> str | None:
    """Choose an existing path without copying files or creating directories."""
    picker = _PathPicker(value, title, suffixes, directory)
    try:
        await picker.navigate(picker.path)
        return await picker
    finally:
        if not picker.is_deleted:
            picker.delete()
