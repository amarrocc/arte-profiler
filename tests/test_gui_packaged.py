"""The packaged browser UI provides an explicit and guarded shutdown action."""

import pytest

pytest.importorskip("nicegui")
pytest.importorskip("pytest_asyncio")

from nicegui import app, ui
from nicegui.testing import User

from arte_profiler.gui import packaged

pytest_plugins = ["nicegui.testing.user_plugin"]
pytestmark = pytest.mark.nicegui_main_file("")


async def test_packaged_page_quit_confirm_cancel_and_confirm(user: User, monkeypatch):
    shutdowns = []
    monkeypatch.setattr(app, "shutdown", lambda: shutdowns.append(True))
    monkeypatch.setattr(packaged.multiprocessing, "active_children", lambda: [])
    ui.page("/")(packaged.packaged_page)
    await user.open("/")
    await user.should_see("Choose a workflow")
    user.find("Quit Arte-Profiler").click()
    await user.should_see("Quit Arte-Profiler?")
    user.find("Keep open").click()
    assert not shutdowns
    user.find("Quit Arte-Profiler").click()
    user.find(marker="confirm-quit").click()
    assert shutdowns == [True]


async def test_quit_does_not_interrupt_processing_even_if_started_after_confirmation(user: User, monkeypatch):
    shutdowns = []
    children = []
    monkeypatch.setattr(app, "shutdown", lambda: shutdowns.append(True))
    monkeypatch.setattr(packaged.multiprocessing, "active_children", lambda: children)
    ui.page("/")(packaged.packaged_page)
    await user.open("/")
    children.append(object())
    user.find("Quit Arte-Profiler").click()
    await user.should_see("Processing is running. Wait for it to finish before quitting.")
    await user.should_not_see(ui.dialog)
    children.clear()
    user.find("Quit Arte-Profiler").click()
    children.append(object())
    user.find(marker="confirm-quit").click()
    assert not shutdowns
    children.clear()
    user.find(marker="confirm-quit").click()
    assert shutdowns == [True]
