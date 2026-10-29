"""Browser-free coverage for the guided GUI and its local filesystem picker."""

from pathlib import Path

import pytest

pytest.importorskip("nicegui")
pytest.importorskip("pytest_asyncio")

from nicegui import ui
from nicegui.testing import User

from arte_profiler.gui.app import create_page

pytest_plugins = ["nicegui.testing.user_plugin"]
pytestmark = pytest.mark.nicegui_main_file("")

ADDITIONAL_CHOICES = {
    "none": "I only have the profiling target.",
    "same": "I have another target in the same TIFF image.",
    "separate": "I have a target in a separate TIFF image.",
}


_DATA = Path()


def _path(name: str) -> str:
    return str(_DATA / name)


@pytest.fixture(autouse=True)
def gui_files_and_preferences(tmp_path_factory, monkeypatch):
    import numpy as np
    import tifffile

    # Existing wizard tests now use real, tiny TIFF files for GUI preflight.
    data = tmp_path_factory.mktemp("gui-inputs")
    monkeypatch.setitem(globals(), "_DATA", data)
    for name in ['another-capture.tiff', 'chosen-output', 'dual.tiff', 'evaluation.cie', 'evaluation.tif', 'evaluation.tiff', 'existing.icc', 'existing.icm', 'measured.cie', 'profiling.cie', 'profiling.tiff', 'results', 'separate.tiff']:
        path = data / name
        if path.suffix in {".tif", ".tiff"}:
            tifffile.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint16), photometric="rgb")
        elif path.suffix in {".cie", ".icc", ".icm"}:
            path.write_text("GUI fixture; no scientific parsing is performed")
    history = tmp_path_factory.mktemp("gui-preferences") / "browse.json"
    preferences_globals = create_page.__globals__["pick_path"].__globals__["last_directory"].__globals__
    monkeypatch.setitem(preferences_globals, "_preferences_path", lambda: history)



async def _start(user: User, existing: bool = False) -> None:
    ui.page("/")(create_page)
    await user.open("/")
    if existing:
        user.find("Evaluate an existing profile").click()
    user.find("Continue").click()
    await user.should_see("Evaluation inputs" if existing else "Profiling inputs")


def _select_target(user: User, prefix: str, target: str) -> None:
    user.find(f"{prefix} color target type").elements.pop().set_value(target)


def _advanced(user: User) -> None:
    user.find("Advanced options").elements.pop().value = True


async def _profiling_inputs(user: User) -> None:
    await _start(user)
    user.find("Profiling TIFF").type(_path("profiling.tiff"))
    _select_target(user, "Profiling", "ColorCheckerSG")
    await user.should_not_see("Results folder")
    await user.should_not_see("An ICC profile and a creation report are always included.")


async def _additional(user: User, choice: str) -> None:
    user.find("Continue").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    user.find(ADDITIONAL_CHOICES[choice]).click()
    await user.should_not_see("Results folder")


async def _review(user: User, button: str = "Continue") -> dict[str, str]:
    user.find(button).click()
    if button == "Continue":
        await user.should_see("Save results")
        if not user.find("Results folder").elements.pop().value:
            user.find("Results folder").type(_path("results"))
        user.find("Review").click()
    await user.should_see(marker="review-summary")
    rows = user.find(marker="review-summary").elements.pop().rows
    return {row["setting"]: row["selection"] for row in rows}


async def test_welcome_has_only_two_workflows(user: User) -> None:
    ui.page("/")(create_page)
    await user.open("/")
    await user.should_see("Arte-Profiler")
    radio = user.find(ui.radio).elements.pop()
    assert list(radio.options.values()) == [
        "Create and evaluate a profile",
        "Evaluate an existing profile",
    ]
    for index, workflow in enumerate(("create", "existing")):
        await user.should_see(marker=f"help-workflow-{workflow}")
        button = user.find(marker=f"help-workflow-{workflow}").elements.pop()
        assert button.parent_slot.parent.parent_slot is radio.slots[f"label-{index}"]
    await user.should_not_see(kind=ui.label, content="Generate and evaluate an ICC color profile using supported targets.")
    user.find(marker="help-workflow-create").click()
    await user.should_see("Generate and evaluate an ICC color profile using supported targets.")
    user.find("Close").click()
    await user.should_see("Supported targets")
    for target in ("ColorCheckerSG", "ColorCheckerPassport", "DT-NGT2"):
        await user.should_see(target)
    user.find("Evaluate an existing profile").click()
    await user.should_see(marker="help-workflow-create")
    user.find(marker="help-workflow-existing").click()
    await user.should_see("Evaluate an existing ICC profile using a supported target.")
    user.find("Close").click()
    await user.should_not_see("Create a profile only")
    await user.should_not_see("Profiling TIFF")
    await user.should_not_see("Existing ICC profile")
    await user.should_not_see("Run")


@pytest.mark.parametrize("choice", ["none", "same", "separate"])
async def test_creation_choices_and_review(user: User, choice: str) -> None:
    await _profiling_inputs(user)
    selector = user.find("Profiling color target type").elements.pop()
    assert set(selector.options) == {"ColorCheckerSG", "ColorCheckerPassport", "DT-NGT2"}
    await user.should_not_see("Profiling target CIE reference values (optional)")
    await user.should_not_see("Evaluation TIFF")
    await _additional(user, choice)

    await user.should_not_see("Profiling TIFF")
    await user.should_see("Step 3 of 5 · Evaluation")
    await user.should_not_see("The creation report primarily verifies")
    recommendation = (
        "Evaluation using a second different target is strongly recommended to reduce "
        "the risk of overestimating color accuracy."
    )
    check_recommendation = user.should_see if choice == "none" else user.should_not_see
    await check_recommendation(recommendation)
    radio = user.find(ui.radio).elements.pop()
    assert radio.options == ADDITIONAL_CHOICES
    if choice == "none":
        await user.should_not_see("Evaluation color target type")
        await user.should_not_see(marker="browse-evaluation_image")
    else:
        await user.should_see("Evaluation color target type")
        await user.should_not_see("Choose a target type different from the profiling target type.")
        help_button = user.find(marker="help-evaluation_target").elements.pop()
        tooltip = next(child for child in help_button.default_slot.children if isinstance(child, ui.tooltip))
        assert "must be different from the profiling target type" in tooltip.text
        await user.should_not_see("Evaluation target CIE reference values (optional)")
        _select_target(user, "Evaluation", "DT-NGT2")
        if choice == "separate":
            await user.should_see(kind=ui.input, content="Evaluation TIFF")
            user.find("Evaluation TIFF").type(_path("evaluation.tif"))
        else:
            await user.should_not_see(marker="browse-evaluation_image")
            await user.should_see(f"Evaluation TIFF: {_path('profiling.tiff')}")

    rows = await _review(user)
    assert rows["Profiling TIFF"] == _path("profiling.tiff")
    assert rows["Profiling target CIE reference values"] == "Chart generic reference values"
    assert rows["Profile algorithm"] == "g — Gamma + matrix (default)"
    assert rows["Profile complexity"] == "m — Medium (default)"
    assert "Creation check" not in rows
    assert "Additional evaluation" not in rows
    assert rows["Evaluation"] == ADDITIONAL_CHOICES[choice]
    assert rows["Results folder"] == _path("results")
    assert "creation report" in rows["Expected outputs"]
    if choice == "none":
        assert "Evaluation TIFF" not in rows
        assert rows["Expected outputs"] == "ICC profile and creation report"
    else:
        assert rows["Evaluation TIFF"] == (
            _path("profiling.tiff") if choice == "same" else _path("evaluation.tif")
        )
        assert rows["Evaluation color target type"] == "DT-NGT2"
        assert rows["Expected outputs"] == "ICC profile, creation report, and evaluation report"
    await user.should_see("Processing log")
    await user.should_see(ui.log)
    await user.should_see("Results")
    await user.should_see("No results yet.")
    assert user.find("Run").elements.pop().enabled


async def test_results_folder_has_its_own_step_and_is_preserved(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "none")
    user.find("Continue").click()
    await user.should_see("Step 4 of 5 · Save results")
    await user.should_not_see("Existing output files with the same names will be replaced.")
    await user.should_not_see("Use Browse to select a folder or create a new one, or enter a folder path.")
    user.find("Review").click()
    await user.should_see("Select a results folder.")
    user.find("Results folder").type(_path("chosen-output"))
    user.find("Back").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    await user.should_not_see("Results folder")
    user.find("Back").click()
    await user.should_see("Profiling inputs")
    await user.should_not_see("Results folder")
    user.find("Continue").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    user.find("Continue").click()
    await user.should_see("Save results")
    assert user.find("Results folder").elements.pop().value == _path("chosen-output")
    rows = await _review(user, "Review")
    await user.should_see("Step 5 of 5 · Review")
    assert rows["Results folder"] == _path("chosen-output")
    await user.should_see("Existing output files with the same names will be replaced.")


async def test_existing_profile_skips_creation_and_additional_choices(user: User) -> None:
    await _start(user, existing=True)
    await user.should_not_see("Profiling TIFF")
    await user.should_not_see("Evaluation target CIE reference values (optional)")
    user.find("Existing ICC profile").type(_path("existing.icc"))
    user.find("Evaluation TIFF").type(_path("evaluation.tiff"))
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    await user.should_see("Evaluation target CIE reference values (optional)")
    user.find("Evaluation target CIE reference values (optional)").type(_path("measured.cie"))
    rows = await _review(user, "Continue")
    assert rows["Existing ICC profile"] == _path("existing.icc")
    assert rows["Evaluation target CIE reference values"] == _path("measured.cie")
    assert rows["Expected outputs"] == "Evaluation report"
    assert "Creation check" not in rows
    assert "Evaluation" not in rows
    await user.should_not_see("Which additional target have you captured for evaluation?")
    user.find("Back").click()
    await user.should_see("Step 3 of 4 · Save results")
    await user.should_see("Results folder")
    user.find("Back").click()
    await user.should_see("Existing ICC profile")
    assert user.find(kind=ui.input, content="Evaluation TIFF").elements.pop().value == _path("evaluation.tiff")


async def test_back_navigation_and_image_choices_preserve_values(user: User) -> None:
    await _profiling_inputs(user)
    _advanced(user)
    await user.should_see("Profiling target CIE reference values (optional)")
    user.find("Profiling target CIE reference values (optional)").type(_path("profiling.cie"))
    await _additional(user, "separate")
    await user.should_see(kind=ui.input, content="Evaluation TIFF")
    user.find("Evaluation TIFF").type(_path("separate.tiff"))
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    await user.should_see("Evaluation target CIE reference values (optional)")
    user.find("Evaluation target CIE reference values (optional)").type(_path("evaluation.cie"))
    await _review(user)
    user.find("Back").click()
    await user.should_see("Save results")
    user.find("Back").click()
    await user.should_see(kind=ui.input, content="Evaluation TIFF")
    assert user.find(kind=ui.input, content="Evaluation TIFF").elements.pop().value == _path("separate.tiff")
    assert user.find("Evaluation target CIE reference values (optional)").elements.pop().value == _path("evaluation.cie")

    user.find(ADDITIONAL_CHOICES["same"]).click()
    await user.should_not_see(marker="browse-evaluation_image")
    user.find("Back").click()
    await user.should_see("Profiling TIFF")
    assert user.find("Profiling target CIE reference values (optional)").elements.pop().value == _path("profiling.cie")
    user.find("Profiling TIFF").clear().type(_path("dual.tiff"))
    user.find("Continue").click()
    await user.should_see(f"Evaluation TIFF: {_path('dual.tiff')}")
    rows = await _review(user)
    assert rows["Evaluation TIFF"] == _path("dual.tiff")
    assert rows["Evaluation target CIE reference values"] == _path("evaluation.cie")
    user.find("Back").click()
    await user.should_see("Save results")
    user.find("Back").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    user.find(ADDITIONAL_CHOICES["separate"]).click()
    await user.should_see(kind=ui.input, content="Evaluation TIFF")
    assert user.find(kind=ui.input, content="Evaluation TIFF").elements.pop().value == _path("separate.tiff")

    user.find("Back").click()
    await user.should_see("Profiling inputs")
    user.find("Back").click()
    await user.should_see("Choose a workflow")
    user.find("Evaluate an existing profile").click()
    user.find("Continue").click()
    await user.should_see("Existing ICC profile")
    user.find("Existing ICC profile").type(_path("existing.icm"))
    rows = await _review(user, "Continue")
    assert rows["Results folder"] == _path("results")
    assert rows["Expected outputs"] == "Evaluation report"
    assert "Profiling TIFF" not in rows


async def test_additional_target_excludes_profiling_type_and_updates_after_back(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "separate")
    await user.should_see(kind=ui.input, content="Evaluation TIFF")
    user.find("Evaluation TIFF").type(_path("another-capture.tiff"))
    selector = user.find("Evaluation color target type").elements.pop()
    assert selector.options == ["ColorCheckerPassport", "DT-NGT2"]
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    user.find("Evaluation target CIE reference values (optional)").type(_path("evaluation.cie"))
    user.find("Back").click()
    await user.should_see("Profiling inputs")
    _select_target(user, "Profiling", "DT-NGT2")
    user.find("Continue").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    selector = user.find("Evaluation color target type").elements.pop()
    assert selector.options == ["ColorCheckerSG", "ColorCheckerPassport"]
    assert selector.value is None
    assert user.find("Evaluation target CIE reference values (optional)").elements.pop().value == ""
    user.find("Back").click()
    await user.should_see("Profiling inputs")
    user.find("Back").click()
    await user.should_see("Choose a workflow")
    user.find("Evaluate an existing profile").click()
    user.find("Continue").click()
    await user.should_see("Existing ICC profile")
    selector = user.find("Evaluation color target type").elements.pop()
    assert set(selector.options) == {"ColorCheckerSG", "ColorCheckerPassport", "DT-NGT2"}


@pytest.mark.parametrize("failed", [False, True])
async def test_run_streams_logs_and_preserves_partial_results(user: User, tmp_path, monkeypatch, failed) -> None:
    from arte_profiler.gui.service import JobEvent

    artifact = tmp_path / "input_profile.icc"
    artifact.write_text("ICC placeholder")
    requests = []

    class Controller:
        running = False

        def start(self, request):
            requests.append(request)
            self.running = True

        def poll(self):
            self.running = False
            return [
                JobEvent("stage", "Generating creation report…"),
                JobEvent("log", "Argyll output received"),
                JobEvent("artifact", "ICC profile", artifact),
                JobEvent("failed" if failed else "done", "Report failed" if failed else "Processing completed."),
            ]

    # NiceGUI's fixture removes page modules from sys.modules between tests;
    # patch the globals of the page function retained by this test module.
    monkeypatch.setitem(create_page.__globals__, "JobController", Controller)
    await _profiling_inputs(user)
    user.find("Advanced profile settings").elements.pop().value = True
    user.find(marker="profile-algorithm").elements.pop().set_value("x")
    user.find(marker="profile-complexity").elements.pop().set_value("h")
    await _additional(user, "none")
    await _review(user)
    user.find("Run").click()
    await user.should_see("Report failed" if failed else "Processing completed.")
    await user.should_see("Argyll output received")
    await user.should_see(str(artifact))
    await user.should_see("Profile algorithm: x — XYZ LUT")
    await user.should_see("Profile complexity: h — High")
    await user.should_see("Download")
    await user.should_not_see("No results yet.")
    downloads = []
    monkeypatch.setattr(type(user.download), "__call__", lambda self, path: downloads.append(path))
    user.find("Download").click()
    assert requests[0].profiling.cie is None
    assert requests[0].evaluation is None
    assert requests[0].existing_icc is None
    assert requests[0].profile_algorithm == "x"
    assert requests[0].profile_complexity == "h"
    # Controls become available after worker cleanup, including failures.
    import asyncio
    await asyncio.sleep(0.15)
    assert downloads == [artifact]
    assert user.find("Run").elements.pop().enabled
    assert user.find("Back").elements.pop().enabled


async def test_run_rechecks_a_file_removed_after_review(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "none")
    await _review(user)
    Path(_path("profiling.tiff")).unlink()
    user.find("Run").click()
    await user.should_see(f"File {_path('profiling.tiff')} not found.")
    assert user.find("Run").elements.pop().enabled


@pytest.mark.parametrize("mode", ["none", "same", "separate", "existing"])
def test_request_snapshot_uses_only_active_inputs(tmp_path, mode) -> None:
    from arte_profiler.gui.app import _FormState

    state = _FormState(
        workflow="existing" if mode == "existing" else "create",
        additional=mode if mode != "existing" else "none",
        profiling_image=str(tmp_path / "profiling.tiff"), profiling_target="ColorCheckerSG",
        profiling_cie="", evaluation_image=str(tmp_path / "evaluation.tiff"),
        evaluation_target="DT-NGT2", evaluation_cie=str(tmp_path / "evaluation.cie"),
        existing_icc=str(tmp_path / "existing.icc"), output_directory=str(tmp_path / "results"),
    )
    request = state.to_request()
    assert request.output_directory == tmp_path / "results"
    if mode == "existing":
        assert request.profiling is None
        assert request.existing_icc == tmp_path / "existing.icc"
    else:
        assert request.profiling.cie is None
        assert request.existing_icc is None
    if mode == "none":
        assert request.evaluation is None
    else:
        assert request.evaluation.image == tmp_path / ("profiling.tiff" if mode == "same" else "evaluation.tiff")
        assert request.evaluation.cie == tmp_path / "evaluation.cie"


async def test_review_requires_inputs_only_for_active_choices(user: User) -> None:
    await _start(user)
    user.find("Continue").click()
    await user.should_see("Select the profiling TIFF image")
    user.find("Profiling TIFF").type(_path("profiling.tiff"))
    _select_target(user, "Profiling", "ColorCheckerSG")
    await _additional(user, "separate")
    user.find("Continue").click()
    await user.should_see("Select the evaluation TIFF image")
    user.find(ADDITIONAL_CHOICES["none"]).click()
    rows = await _review(user)
    assert "Evaluation TIFF" not in rows
    assert rows["Expected outputs"] == "ICC profile and creation report"


@pytest.mark.parametrize("name,content,expected", [
    ("wrong.jpg", b"not TIFF", "unsupported file extension (.jpg)"),
    ("missing.tiff", None, "file not found"),
    ("broken.tiff", b"not TIFF", "invalid or unreadable TIFF"),
])
async def test_continue_rejects_wrong_missing_or_invalid_tiff(user: User, tmp_path, name, content, expected):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    await _start(user)
    user.find("Profiling TIFF").type(str(path))
    _select_target(user, "Profiling", "ColorCheckerSG")
    user.find("Continue").click()
    await user.should_see(expected)
    await user.should_see("Profiling inputs")
    await user.should_not_see("Which additional target have you captured for evaluation?")


async def test_continue_checks_optional_cie_and_existing_icc_before_advancing(user: User):
    await _profiling_inputs(user)
    _advanced(user)
    user.find("Profiling target CIE reference values (optional)").type(_path("missing.cie"))
    user.find("Continue").click()
    await user.should_see("Profiling CIE reference: file not found")
    user.find("Back").click()
    await user.should_see("Choose a workflow")
    user.find("Evaluate an existing profile").click()
    user.find("Continue").click()
    await user.should_see("Existing ICC profile")
    user.find("Existing ICC profile").type(_path("missing.icc"))
    user.find("Continue").click()
    await user.should_see("Existing ICC profile: file not found")
    await user.should_not_see("Save results")


async def test_back_navigation_during_slow_preflight_does_not_advance_later(user: User, monkeypatch):
    import asyncio
    from threading import Event

    started, release = Event(), Event()

    def slow_check(*args):
        started.set()
        release.wait(timeout=5)
        return ""

    await _profiling_inputs(user)
    monkeypatch.setitem(create_page.__globals__, "_chart_error", slow_check)
    user.find("Continue").click()
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await user.should_see("Checking selected files…")
        user.find("Back").click()
        await user.should_see("Choose a workflow")
    finally:
        release.set()
    await asyncio.sleep(0.15)
    await user.should_see("Choose a workflow")
    await user.should_not_see("Which additional target have you captured for evaluation?")


@pytest.mark.parametrize(
    "label,attribute,suffix,context",
    [
        ("Profiling TIFF", "profiling_image", ".TIFF", "profiling"),
        ("Evaluation TIFF", "evaluation_image", ".tif", "additional"),
        ("Profiling target CIE reference values (optional)", "profiling_cie", ".CIE", "profiling"),
        ("Evaluation target CIE reference values (optional)", "evaluation_cie", ".cie", "additional"),
        ("Existing ICC profile", "existing_icc", ".icm", "existing"),
    ],
)
async def test_browse_selects_matching_file(
    user: User, tmp_path: Path, label: str, attribute: str, suffix: str, context: str
) -> None:
    selected = tmp_path / f"référence capture{suffix}"
    selected.write_text("GUI test fixture")
    (tmp_path / "unrelated.txt").write_text("Not a supported file")
    (tmp_path / "subfolder").mkdir()
    if context == "additional":
        await _profiling_inputs(user)
        await _additional(user, "separate")
        await user.should_see(kind=ui.input, content="Evaluation TIFF")
    else:
        await _start(user, existing=context == "existing")
    if suffix.lower() == ".cie":
        _advanced(user)
        await user.should_see(label)
        await user.should_not_see("Select a .cie file containing chart-specific measurements,")
    user.find(kind=ui.input, content=label).type(str(tmp_path))
    user.find(marker=f"browse-{attribute}").click()
    await user.should_see(f"Browse: {label}")
    table = user.find(ui.table).elements.pop()
    assert [row["name"] for row in table.rows] == ["subfolder", selected.name]
    row = next(row for row in table.rows if row["name"] == selected.name)
    user.find(ui.table).trigger("rowClick", [{}, row, 1])
    await user.should_not_see(ui.dialog)
    assert user.find(kind=ui.input, content=label).elements.pop().value == str(selected)


async def test_browse_directory_navigation_and_cancel(user: User, tmp_path: Path) -> None:
    folder = tmp_path / "output folder"
    folder.mkdir()
    (tmp_path / "not-a-directory.tiff").write_text("GUI test fixture")
    await _profiling_inputs(user)
    await _additional(user, "none")
    user.find("Continue").click()
    await user.should_see("Save results")
    user.find("Results folder").type(str(tmp_path))
    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Results folder")
    table = user.find(ui.table).elements.pop()
    assert [row["name"] for row in table.rows] == [folder.name]
    user.find(ui.table).trigger("rowClick", [{}, table.rows[0], 0])
    await user.should_see(content=str(folder), kind=ui.input)
    user.find("Select this folder").click()
    await user.should_not_see(ui.dialog)
    assert user.find("Results folder").elements.pop().value == str(folder)

    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Results folder")
    user.find("Up").click()
    await user.should_see(content=str(tmp_path), kind=ui.input)
    user.find("Location").clear().type(str(tmp_path / "missing"))
    user.find("Location").trigger("keydown.enter")
    await user.should_see("Cannot open folder:")
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    assert user.find("Results folder").elements.pop().value == str(folder)


async def test_browse_remembers_cancelled_navigation_and_shares_only_input_history(user: User, tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    visited = root / "visited"
    visited.mkdir(parents=True)
    results = tmp_path / "results"
    results.mkdir()
    await _start(user)
    user.find("Profiling TIFF").type(str(root))
    user.find(marker="browse-profiling_image").click()
    await user.should_see("Browse: Profiling TIFF")
    await user.should_not_see("Go")
    table = user.find(ui.table).elements.pop()
    user.find(ui.table).trigger("rowClick", [{}, table.rows[0], 0])
    await user.should_see(content=str(visited), kind=ui.input)
    await user.should_see("No data available")
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    user.find(marker="browse-profiling_image").click()
    await user.should_see(content=str(visited), kind=ui.input)
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    _advanced(user)
    user.find(marker="browse-profiling_cie").click()
    await user.should_see(content=str(visited), kind=ui.input)
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    user.find("Profiling TIFF").clear().type(_path("profiling.tiff"))
    _select_target(user, "Profiling", "ColorCheckerSG")
    await _additional(user, "none")
    user.find("Continue").click()
    await user.should_see("Save results")
    user.find("Results folder").type(str(results))
    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Results folder")
    await user.should_see(content=str(results), kind=ui.input)
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    history = create_page.__globals__["pick_path"].__globals__["last_directory"]
    assert history(False) == str(visited)
    assert history(True) == str(results)


async def test_unavailable_remembered_directory_falls_back_home(user: User, tmp_path: Path) -> None:
    picker = create_page.__globals__["pick_path"].__globals__
    picker["remember_directory"](tmp_path / "missing", False)
    assert picker["_initial_directory"]("", False) == Path.home()


async def _browse_results_folder(user: User, path: Path) -> None:
    await _profiling_inputs(user)
    await _additional(user, "none")
    user.find("Continue").click()
    await user.should_see("Save results")
    user.find("Results folder").type(str(path))
    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Results folder")
    await user.should_see(content=str(path), kind=ui.input)


async def test_create_new_results_folder_and_select_it(user: User, tmp_path: Path) -> None:
    await _browse_results_folder(user, tmp_path)
    user.find("New folder").click()
    await user.should_see("New folder name")
    user.find("New folder name").type("Référence results")
    user.find("Create").click()
    folder = tmp_path / "Référence results"
    await user.should_see(content=str(folder), kind=ui.input)
    assert folder.is_dir()
    await user.should_not_see("New folder name")
    user.find("Select this folder").click()
    await user.should_not_see(ui.dialog)
    assert user.find("Results folder").elements.pop().value == str(folder)
    rows = await _review(user, "Review")
    assert rows["Results folder"] == str(folder)


@pytest.mark.parametrize("name", ["", "..", "../outside", "nested/folder", r"..\outside", "C:outside"])
async def test_new_folder_rejects_invalid_names(user: User, tmp_path: Path, name: str) -> None:
    await _browse_results_folder(user, tmp_path)
    user.find("New folder").click()
    await user.should_see("New folder name")
    if name:
        user.find("New folder name").type(name)
    user.find("Create").click()
    await user.should_see("Cannot create folder:")
    assert list(tmp_path.iterdir()) == []
    assert user.find("Location").elements.pop().value == str(tmp_path)


@pytest.mark.parametrize("existing_file", [False, True])
async def test_new_folder_does_not_replace_existing_path(user: User, tmp_path: Path, existing_file: bool) -> None:
    existing = tmp_path / "existing"
    if existing_file:
        existing.write_text("Keep this file")
    else:
        existing.mkdir()
    await _browse_results_folder(user, tmp_path)
    user.find("New folder").click()
    await user.should_see("New folder name")
    user.find("New folder name").type("existing")
    user.find("Create").click()
    await user.should_see("A file or folder with this name already exists.")
    assert user.find("Location").elements.pop().value == str(tmp_path)
    assert existing.read_text() == "Keep this file" if existing_file else existing.is_dir()


async def test_folder_permission_error_is_shown(user: User, tmp_path: Path, monkeypatch) -> None:
    def denied(parent, name):
        raise PermissionError("Write access denied")

    picker_globals = create_page.__globals__["pick_path"].__globals__
    monkeypatch.setitem(picker_globals, "_make_directory", denied)
    await _browse_results_folder(user, tmp_path)
    user.find("New folder").click()
    await user.should_see("New folder name")
    user.find("New folder name").type("results")
    user.find("Create").click()
    await user.should_see("Cannot create folder: Write access denied")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("attribute,context", [
    ("profiling_image", "profiling"), ("profiling_target", "profiling"),
    ("profiling_cie", "advanced"), ("existing_icc", "existing"),
    ("evaluation_image", "existing"), ("evaluation_target", "existing"),
    ("evaluation_cie", "existing_advanced"), ("output_directory", "save"),
])
async def test_field_help_is_accessible_and_opens_on_click(user: User, attribute: str, context: str) -> None:
    if context.startswith("existing"):
        await _start(user, existing=True)
    else:
        await _profiling_inputs(user)
    if "advanced" in context:
        _advanced(user)
    if context == "save":
        await _additional(user, "none")
        user.find("Continue").click()
        await user.should_see("Save results")
    button = user.find(marker=f"help-{attribute}").elements.pop()
    assert button._props["aria-label"].startswith("About ")
    tooltip = next(child for child in button.default_slot.children if isinstance(child, ui.tooltip))
    assert tooltip.text
    if attribute == "existing_icc":
        assert tooltip.text == "Select the existing ICC profile whose color accuracy you want to evaluate."
    if attribute == "evaluation_target":
        assert "must be different" not in tooltip.text
    assert tooltip._style["font-size"] == "14px"
    assert tooltip._style["line-height"] == "1.5"
    user.find(marker=f"help-{attribute}").trigger("focusin")
    user.find(marker=f"help-{attribute}").trigger("focusout")
    user.find(marker=f"help-{attribute}").click()
    await user.should_see(ui.dialog)
    await user.should_see(tooltip.text)
    user.find("Close").click()
    await user.should_not_see(ui.dialog)
