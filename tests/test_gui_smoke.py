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
    user.find(ui.expansion).elements.pop().value = True


async def _profiling_inputs(user: User) -> None:
    await _start(user)
    user.find("Profiling TIFF").type("/data/profiling.tiff")
    _select_target(user, "Profiling", "ColorCheckerSG")
    await user.should_not_see("Output directory")
    await user.should_not_see("An ICC profile and a creation report are always included.")


async def _additional(user: User, choice: str, fill_output: bool = True) -> None:
    user.find("Continue").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    user.find(ADDITIONAL_CHOICES[choice]).click()
    await user.should_see("Output directory")
    if fill_output and not user.find("Output directory").elements.pop().value:
        user.find("Output directory").type("/data/results")


async def _review(user: User, button: str = "Review") -> dict[str, str]:
    user.find(button).click()
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
    await user.should_see("Generate and evaluate an ICC color profile using supported targets.")
    await user.should_see("Supported targets")
    for target in ("ColorCheckerSG", "ColorCheckerPassport", "DT-NGT2"):
        await user.should_see(target)
    user.find("Evaluate an existing profile").click()
    await user.should_see("Evaluate an existing ICC profile using a supported target.")
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
    await user.should_see("Step 3 of 4 · Evaluation")
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
        await user.should_not_see("Evaluation target CIE reference values (optional)")
        _select_target(user, "Evaluation", "DT-NGT2")
        if choice == "separate":
            await user.should_see("Evaluation TIFF")
            user.find("Evaluation TIFF").type("/data/evaluation.tif")
        else:
            await user.should_not_see(marker="browse-evaluation_image")
            await user.should_see("Reusing profiling TIFF: /data/profiling.tiff")

    rows = await _review(user)
    assert rows["Profiling TIFF"] == "/data/profiling.tiff"
    assert rows["Profiling target CIE reference values"] == "Chart generic reference values"
    assert "Creation check" not in rows
    assert "Additional evaluation" not in rows
    assert rows["Evaluation"] == ADDITIONAL_CHOICES[choice]
    assert rows["Output directory"] == "/data/results"
    assert "creation report" in rows["Expected outputs"]
    if choice == "none":
        assert "Evaluation TIFF" not in rows
        assert rows["Expected outputs"] == "ICC profile and creation report"
    else:
        assert rows["Evaluation TIFF"] == (
            "/data/profiling.tiff" if choice == "same" else "/data/evaluation.tif"
        )
        assert rows["Evaluation color target type"] == "DT-NGT2"
        assert "additional evaluation report" in rows["Expected outputs"]
    await user.should_see("Processing log")
    await user.should_see(ui.log)
    await user.should_see("Results")
    await user.should_see("No results yet.")
    assert user.find("Run").elements.pop().enabled


async def test_output_directory_is_selected_at_evaluation_and_preserved(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "none", fill_output=False)
    user.find("Review").click()
    await user.should_see("Select an output directory.")
    user.find("Output directory").type("/data/chosen-output")
    user.find("Back").click()
    await user.should_see("Profiling inputs")
    await user.should_not_see("Output directory")
    user.find("Continue").click()
    await user.should_see("Output directory")
    assert user.find("Output directory").elements.pop().value == "/data/chosen-output"
    rows = await _review(user)
    assert rows["Output directory"] == "/data/chosen-output"


async def test_existing_profile_skips_creation_and_additional_choices(user: User) -> None:
    await _start(user, existing=True)
    await user.should_not_see("Profiling TIFF")
    await user.should_not_see("Evaluation target CIE reference values (optional)")
    user.find("Existing ICC profile").type("/data/existing.icc")
    user.find("Evaluation TIFF").type("/data/evaluation.tiff")
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    await user.should_see("Evaluation target CIE reference values (optional)")
    user.find("Evaluation target CIE reference values (optional)").type("/data/measured.cie")
    user.find("Output directory").type("/data/results")
    rows = await _review(user, "Continue")
    assert rows["Existing ICC profile"] == "/data/existing.icc"
    assert rows["Evaluation target CIE reference values"] == "/data/measured.cie"
    assert rows["Expected outputs"] == "Evaluation report"
    assert "Creation check" not in rows
    assert "Evaluation" not in rows
    await user.should_not_see("Which additional target have you captured for evaluation?")
    user.find("Back").click()
    await user.should_see("Existing ICC profile")
    assert user.find("Evaluation TIFF").elements.pop().value == "/data/evaluation.tiff"


async def test_back_navigation_and_image_choices_preserve_values(user: User) -> None:
    await _profiling_inputs(user)
    _advanced(user)
    await user.should_see("Profiling target CIE reference values (optional)")
    user.find("Profiling target CIE reference values (optional)").type("/data/profiling.cie")
    await _additional(user, "separate")
    await user.should_see("Evaluation TIFF")
    user.find("Evaluation TIFF").type("/data/separate.tiff")
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    await user.should_see("Evaluation target CIE reference values (optional)")
    user.find("Evaluation target CIE reference values (optional)").type("/data/evaluation.cie")
    await _review(user)
    user.find("Back").click()
    await user.should_see("Evaluation TIFF")
    assert user.find("Evaluation TIFF").elements.pop().value == "/data/separate.tiff"
    assert user.find("Evaluation target CIE reference values (optional)").elements.pop().value == "/data/evaluation.cie"

    user.find(ADDITIONAL_CHOICES["same"]).click()
    await user.should_not_see(marker="browse-evaluation_image")
    user.find("Back").click()
    await user.should_see("Profiling TIFF")
    assert user.find("Profiling target CIE reference values (optional)").elements.pop().value == "/data/profiling.cie"
    user.find("Profiling TIFF").clear().type("/data/dual.tiff")
    user.find("Continue").click()
    await user.should_see("Reusing profiling TIFF: /data/dual.tiff")
    rows = await _review(user)
    assert rows["Evaluation TIFF"] == "/data/dual.tiff"
    assert rows["Evaluation target CIE reference values"] == "/data/evaluation.cie"
    user.find("Back").click()
    await user.should_see("Which additional target have you captured for evaluation?")
    user.find(ADDITIONAL_CHOICES["separate"]).click()
    await user.should_see("Evaluation TIFF")
    assert user.find("Evaluation TIFF").elements.pop().value == "/data/separate.tiff"

    user.find("Back").click()
    await user.should_see("Profiling inputs")
    user.find("Back").click()
    await user.should_see("Choose a workflow")
    user.find("Evaluate an existing profile").click()
    user.find("Continue").click()
    await user.should_see("Existing ICC profile")
    user.find("Existing ICC profile").type("/data/existing.icm")
    assert user.find("Output directory").elements.pop().value == "/data/results"
    rows = await _review(user, "Continue")
    assert rows["Expected outputs"] == "Evaluation report"
    assert "Profiling TIFF" not in rows


async def test_additional_target_excludes_profiling_type_and_updates_after_back(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "separate")
    await user.should_see("Evaluation TIFF")
    user.find("Evaluation TIFF").type("/data/another-capture.tiff")
    selector = user.find("Evaluation color target type").elements.pop()
    assert selector.options == ["ColorCheckerPassport", "DT-NGT2"]
    _select_target(user, "Evaluation", "DT-NGT2")
    _advanced(user)
    user.find("Evaluation target CIE reference values (optional)").type("/data/evaluation.cie")
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
    await _additional(user, "none")
    await _review(user)
    user.find("Run").click()
    await user.should_see("Report failed" if failed else "Processing completed.")
    await user.should_see("Argyll output received")
    await user.should_see(str(artifact))
    await user.should_see("Download")
    await user.should_not_see("No results yet.")
    downloads = []
    monkeypatch.setattr(type(user.download), "__call__", lambda self, path: downloads.append(path))
    user.find("Download").click()
    assert requests[0].profiling.cie is None
    assert requests[0].evaluation is None
    assert requests[0].existing_icc is None
    # Controls become available after worker cleanup, including failures.
    import asyncio
    await asyncio.sleep(0.15)
    assert downloads == [artifact]
    assert user.find("Run").elements.pop().enabled
    assert user.find("Back").elements.pop().enabled


async def test_run_missing_file_shows_error_without_starting_worker(user: User) -> None:
    await _profiling_inputs(user)
    await _additional(user, "none")
    await _review(user)
    user.find("Run").click()
    await user.should_see("File /data/profiling.tiff not found.")
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
    user.find("Profiling TIFF").type("/data/profiling.tiff")
    _select_target(user, "Profiling", "ColorCheckerSG")
    await _additional(user, "separate")
    user.find("Review").click()
    await user.should_see("Select the evaluation TIFF image")
    user.find(ADDITIONAL_CHOICES["none"]).click()
    rows = await _review(user)
    assert "Evaluation TIFF" not in rows
    assert rows["Expected outputs"] == "ICC profile and creation report"


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
        await user.should_see("Evaluation TIFF")
    else:
        await _start(user, existing=context == "existing")
    if suffix.lower() == ".cie":
        _advanced(user)
        await user.should_see(label)
        await user.should_see("chart-specific measurements")
        await user.should_see("leave blank to use chart generic reference values.")
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
    await _additional(user, "none", fill_output=False)
    user.find("Output directory").type(str(tmp_path))
    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Output directory")
    table = user.find(ui.table).elements.pop()
    assert [row["name"] for row in table.rows] == [folder.name]
    user.find(ui.table).trigger("rowClick", [{}, table.rows[0], 0])
    await user.should_see(content=str(folder), kind=ui.input)
    user.find("Select this folder").click()
    await user.should_not_see(ui.dialog)
    assert user.find("Output directory").elements.pop().value == str(folder)
    user.find(marker="browse-output_directory").click()
    await user.should_see("Browse: Output directory")
    user.find("Up").click()
    await user.should_see(content=str(tmp_path), kind=ui.input)
    user.find("Location").clear().type(str(tmp_path / "missing"))
    user.find("Go").click()
    await user.should_see("Cannot open folder:")
    user.find("Cancel").click()
    await user.should_not_see(ui.dialog)
    assert user.find("Output directory").elements.pop().value == str(folder)
