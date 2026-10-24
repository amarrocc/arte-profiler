"""Guided browser forms for Arte-Profiler; processing is not connected yet."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml
from nicegui import ui

from .file_picker import pick_path

WORKFLOWS = {
    "create": "Create and evaluate a profile",
    "existing": "Evaluate an existing profile",
}
ADDITIONAL_EVALUATION = {
    "none": "I only have the profiling target.",
    "same": "I have another target in the same TIFF image.",
    "separate": "I have a target in a separate TIFF image.",
}
EVALUATION_RECOMMENDATION = (
    "Evaluation using a second different target is strongly recommended to reduce "
    "the risk of overestimating color accuracy."
)
SAME_PATCHES_NOTE = (
    "This target type has the same patch layout as the profiling target. "
    "A separate TIFF does not by itself provide an independent evaluation across colors."
)


@dataclass
class _FormState:
    workflow: str = "create"
    step: str = "welcome"
    additional: str = "none"
    profiling_image: str = ""
    profiling_target: str | None = None
    profiling_cie: str = ""
    evaluation_image: str = ""
    evaluation_target: str | None = None
    evaluation_cie: str = ""
    existing_icc: str = ""
    output_directory: str = ""
    profiling_advanced: bool = False
    evaluation_advanced: bool = False
    input_error: str = ""
    additional_error: str = ""

    @property
    def is_creation(self) -> bool:
        return self.workflow == "create"

    @property
    def has_additional(self) -> bool:
        return self.is_creation and self.additional != "none"

    @property
    def needs_separate_image(self) -> bool:
        return self.has_additional and self.additional == "separate"

    @property
    def reuses_profiling_image(self) -> bool:
        return self.has_additional and self.additional == "same"

    @property
    def effective_evaluation_image(self) -> str:
        return self.profiling_image if self.reuses_profiling_image else self.evaluation_image

    @property
    def same_patch_layout(self) -> bool:
        return bool(
            self.has_additional and self.profiling_target
            and self.profiling_target == self.evaluation_target
        )

    @property
    def step_label(self) -> str:
        steps = ["welcome", "inputs", "additional", "review"] if self.is_creation else [
            "welcome", "inputs", "review"
        ]
        titles = {
            "welcome": "Choose a workflow",
            "inputs": "Profiling inputs" if self.is_creation else "Evaluation inputs",
            "additional": "Evaluation",
            "review": "Review",
        }
        return f"Step {steps.index(self.step) + 1} of {len(steps)} · {titles[self.step]}"

    @property
    def expected_outputs(self) -> str:
        if not self.is_creation:
            return "Evaluation report"
        if self.has_additional:
            return "ICC profile, creation report, and additional evaluation report"
        return "ICC profile and creation report"

    def review_rows(self) -> list[dict[str, str]]:
        selections = [("Workflow", WORKFLOWS[self.workflow])]
        if self.is_creation:
            selections.extend([
                ("Profiling TIFF", self.profiling_image),
                ("Profiling color target type", self.profiling_target),
                ("Profiling target CIE reference values", self.profiling_cie or "Chart generic reference values"),
                ("Creation check", "Always included: verification on the profiling target"),
                ("Additional evaluation", ADDITIONAL_EVALUATION[self.additional]),
            ])
        else:
            selections.append(("Existing ICC profile", self.existing_icc))
        if not self.is_creation or self.has_additional:
            selections.extend([
                ("Evaluation TIFF", self.effective_evaluation_image),
                ("Evaluation color target type", self.evaluation_target),
                ("Evaluation target CIE reference values", self.evaluation_cie or "Chart generic reference values"),
            ])
        selections.extend([
            ("Output directory", self.output_directory),
            ("Expected outputs", self.expected_outputs),
        ])
        return [{"setting": setting, "selection": str(value)} for setting, value in selections]


def _target_options() -> list[str]:
    manifest = files("arte_profiler") / "data" / "targets" / "targets_manifest.yaml"
    with manifest.open("r", encoding="utf-8") as stream:
        return list(yaml.safe_load(stream))


def _path_field(
    label: str, state: _FormState, attribute: str, *,
    suffixes: tuple[str, ...] = (), directory: bool = False,
) -> ui.row:
    async def browse() -> None:
        result = await pick_path(
            getattr(state, attribute), title=label, suffixes=suffixes, directory=directory,
        )
        if result is not None:
            field.value = result

    with ui.row().classes("w-full flex-nowrap items-center gap-2") as row:
        field = ui.input(
            label, placeholder="Folder path" if directory else "File path",
        ).props("outlined").classes("flex-1 min-w-0").bind_value(state, attribute)
        ui.button("Browse", icon="folder_open", on_click=browse).props("outline").mark(
            f"browse-{attribute}"
        ).tooltip(f"Browse for {label.lower()}")
    return row


def _target_fields(prefix: str, state: _FormState, attribute: str, targets: list[str]) -> None:
    ui.select(targets, label=f"{prefix} color target type").props("outlined").classes(
        "w-full"
    ).bind_value(state, f"{attribute}_target")
    with ui.expansion("Advanced options", icon="tune").classes("w-full").bind_value(
        state, f"{attribute}_advanced"
    ) as advanced:
        with ui.column().classes("w-full gap-3").bind_visibility_from(advanced, "value"):
            _path_field(
                f"{prefix} target CIE reference values (optional)", state, f"{attribute}_cie", suffixes=(".cie",),
            )
            ui.label(
                "Select a .cie file containing chart-specific measurements, "
                "or leave blank to use chart generic reference values."
            ).classes("text-sm text-slate-600")


def _chart_error(prefix: str, image: str, target: str | None, cie: str, targets: list[str]) -> str:
    if not image or Path(image).suffix.lower() not in {".tif", ".tiff"}:
        return f"Select the {prefix.lower()} TIFF image (.tif or .tiff)."
    if target not in targets:
        return f"Select the {prefix.lower()} color target type."
    if cie and Path(cie).suffix.lower() != ".cie":
        return f"The optional {prefix.lower()} reference must be a .cie file."
    return ""


def create_page() -> None:
    """Build a wizard with persistent values for each browser session."""
    state = _FormState()
    targets = _target_options()
    card_classes = "w-full gap-4 p-6 shadow-sm border border-slate-200"

    def go_to(step: str) -> None:
        state.input_error = ""
        state.additional_error = ""
        if step == "review":
            review.rows = state.review_rows()
        state.step = step

    def continue_inputs() -> None:
        if state.is_creation:
            error = _chart_error(
                "Profiling", state.profiling_image, state.profiling_target, state.profiling_cie, targets,
            )
        else:
            error = "" if state.existing_icc and Path(state.existing_icc).suffix.lower() in {
                ".icc", ".icm"
            } else "Select an existing ICC profile (.icc or .icm)."
            error = error or _chart_error(
                "Evaluation", state.evaluation_image, state.evaluation_target, state.evaluation_cie, targets,
            )
            error = error or ("" if state.output_directory and state.output_directory.strip() else "Select an output directory.")
        if error:
            state.input_error = error
        else:
            go_to("additional" if state.is_creation else "review")

    def review_creation() -> None:
        error = _chart_error(
            "Evaluation", state.effective_evaluation_image, state.evaluation_target, state.evaluation_cie, targets,
        ) if state.has_additional else ""
        error = error or ("" if state.output_directory and state.output_directory.strip() else "Select an output directory.")
        if error:
            state.additional_error = error
        else:
            go_to("review")

    ui.colors(primary="#245c66")
    with ui.column().classes("w-full max-w-4xl mx-auto gap-5 p-4 sm:p-8"):
        with ui.column().classes("gap-1"):
            ui.label("Arte-Profiler").classes("text-3xl font-semibold text-slate-900")
            ui.label("Camera profiling and color accuracy evaluation").classes("text-base text-slate-600")
        ui.label().bind_text_from(state, "step_label").classes("text-sm font-medium text-slate-600")

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="welcome"):
            ui.label("Choose a workflow").classes("text-xl font-medium")
            ui.radio(WORKFLOWS, value=state.workflow).bind_value(state, "workflow").classes("w-full")
            ui.label().bind_text_from(
                state, "is_creation", backward=lambda creating:
                "Generate and evaluate an ICC color profile using supported targets."
                if creating else "Evaluate an existing ICC profile using a supported target."
            ).classes("text-slate-600")
            ui.label("Supported targets").classes("font-medium")
            ui.label(", ".join(targets)).classes("text-sm text-slate-600")
            ui.button("Continue", on_click=lambda: go_to("inputs"))

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="inputs"):
            with ui.column().classes("w-full gap-4").bind_visibility_from(state, "is_creation"):
                ui.label("Profiling inputs").classes("text-xl font-medium")
                _path_field("Profiling TIFF", state, "profiling_image", suffixes=(".tif", ".tiff"))
                _target_fields("Profiling", state, "profiling", targets)
            with ui.column().classes("w-full gap-4").bind_visibility_from(
                state, "is_creation", backward=lambda value: not value
            ):
                ui.label("Evaluation inputs").classes("text-xl font-medium")
                _path_field("Existing ICC profile", state, "existing_icc", suffixes=(".icc", ".icm"))
                _path_field("Evaluation TIFF", state, "evaluation_image", suffixes=(".tif", ".tiff"))
                _target_fields("Evaluation", state, "evaluation", targets)
                _path_field("Output directory", state, "output_directory", directory=True)
            ui.label().bind_text_from(state, "input_error").classes("text-sm text-red-800")
            with ui.row().classes("w-full justify-between"):
                ui.button("Back", on_click=lambda: go_to("welcome")).props("outline")
                ui.button("Continue", on_click=continue_inputs)

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="additional"):
            ui.label("Evaluation").classes("text-xl font-medium")
            ui.label("Which additional target have you captured for evaluation?").classes("font-medium")
            ui.radio(ADDITIONAL_EVALUATION, value=state.additional).bind_value(
                state, "additional"
            ).classes("w-full")
            ui.label(EVALUATION_RECOMMENDATION).classes(
                "text-sm text-slate-600 leading-relaxed"
            ).bind_visibility_from(state, "has_additional", backward=lambda value: not value)
            with ui.column().classes("w-full gap-4").bind_visibility_from(state, "has_additional"):
                ui.label().bind_text_from(
                    state, "profiling_image", backward=lambda path: f"Reusing profiling TIFF: {path}"
                ).classes("text-sm text-slate-600 break-all").bind_visibility_from(state, "reuses_profiling_image")
                _path_field(
                    "Evaluation TIFF", state, "evaluation_image", suffixes=(".tif", ".tiff"),
                ).bind_visibility_from(state, "needs_separate_image")
                _target_fields("Evaluation", state, "evaluation", targets)
                ui.label(SAME_PATCHES_NOTE).classes(
                    "text-sm text-amber-900 bg-amber-50 p-3 rounded"
                ).bind_visibility_from(state, "same_patch_layout")
            _path_field("Output directory", state, "output_directory", directory=True)
            ui.label().bind_text_from(state, "additional_error").classes("text-sm text-red-800")
            with ui.row().classes("w-full justify-between"):
                ui.button("Back", on_click=lambda: go_to("inputs")).props("outline")
                ui.button("Review", on_click=review_creation)

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="review"):
            ui.label("Review").classes("text-xl font-medium")
            ui.label("Check the selected inputs and expected outputs before processing.").classes("text-slate-600")
            review = ui.table(
                columns=[
                    {"name": "setting", "label": "Setting", "field": "setting"},
                    {"name": "selection", "label": "Selection", "field": "selection"},
                ],
                column_defaults={
                    "align": "left",
                    "style": "white-space: normal; overflow-wrap: anywhere;",
                },
                rows=[],
                row_key="setting",
            ).props("flat bordered hide-bottom").classes("w-full").mark("review-summary")
            ui.label(SAME_PATCHES_NOTE).classes(
                "text-sm text-amber-900 bg-amber-50 p-3 rounded"
            ).bind_visibility_from(state, "same_patch_layout")
            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.button(
                    "Back", on_click=lambda: go_to("additional" if state.is_creation else "inputs")
                ).props("outline")
                ui.button("Run", icon="play_arrow").disable()
            ui.label("Processing is not connected yet.").classes("text-sm text-slate-600")
            ui.separator()
            ui.label("Processing log").classes("text-lg font-medium")
            ui.log(max_lines=500).classes("w-full h-32 border border-slate-200 rounded")
            ui.label("No processing messages yet.").classes("text-sm text-slate-600")
            ui.label("Results").classes("text-lg font-medium")
            ui.label("No results yet.").classes("text-sm text-slate-600")
