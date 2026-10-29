"""Guided browser forms and processing controls for Arte-Profiler."""

import asyncio
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from nicegui import background_tasks, run, ui

from .controller import JobController
from .file_picker import pick_path
from .packaged import protect_bundle_output
from .service import (
    DEFAULT_PROFILE_ALGORITHM, DEFAULT_PROFILE_COMPLEXITY,
    PROFILE_ALGORITHM_LABELS, PROFILE_COMPLEXITY_LABELS,
    ChartInput, ProcessingRequest, supported_targets,
)
from .validation import file_error

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

# Official GitHub Invertocat from https://brand.github.com/GitHub_Logos.zip.
# Embedded so the footer needs neither a brand icon font nor a network request.
_GITHUB_MARK_SVG = '''<svg width="16" height="16" viewBox="0 0 98 96" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
<path d="M41.4395 69.3848C28.8066 67.8535 19.9062 58.7617 19.9062 46.9902C19.9062 42.2051 21.6289 37.0371 24.5 33.5918C23.2559 30.4336 23.4473 23.7344 24.8828 20.959C28.7109 20.4805 33.8789 22.4902 36.9414 25.2656C40.5781 24.1172 44.4062 23.543 49.0957 23.543C53.7852 23.543 57.6133 24.1172 61.0586 25.1699C64.0254 22.4902 69.2891 20.4805 73.1172 20.959C74.457 23.543 74.6484 30.2422 73.4043 33.4961C76.4668 37.1328 78.0937 42.0137 78.0937 46.9902C78.0937 58.7617 69.1934 67.6621 56.3691 69.2891C59.623 71.3945 61.8242 75.9883 61.8242 81.252L61.8242 91.2051C61.8242 94.0762 64.2168 95.7031 67.0879 94.5547C84.4102 87.9512 98 70.6289 98 49.1914C98 22.1074 75.9883 6.69539e-07 48.9043 4.309e-07C21.8203 1.92261e-07 -1.9479e-07 22.1074 -4.3343e-07 49.1914C-6.20631e-07 70.4375 13.4941 88.0469 31.6777 94.6504C34.2617 95.6074 36.75 93.8848 36.75 91.3008L36.75 83.6445C35.4102 84.2188 33.6875 84.6016 32.1562 84.6016C25.8398 84.6016 22.1074 81.1563 19.4277 74.7441C18.375 72.1602 17.2266 70.6289 15.0254 70.3418C13.877 70.2461 13.4941 69.7676 13.4941 69.1934C13.4941 68.0449 15.4082 67.1836 17.3223 67.1836C20.0977 67.1836 22.4902 68.9063 24.9785 72.4473C26.8926 75.2227 28.9023 76.4668 31.2949 76.4668C33.6875 76.4668 35.2187 75.6055 37.4199 73.4043C39.0469 71.7773 40.291 70.3418 41.4395 69.3848Z" fill="black"/>
</svg>'''


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
    profile_settings_advanced: bool = False
    profile_algorithm: str = DEFAULT_PROFILE_ALGORITHM
    profile_complexity: str = DEFAULT_PROFILE_COMPLEXITY
    evaluation_advanced: bool = False
    input_error: str = ""
    additional_error: str = ""
    save_error: str = ""
    checking_files: bool = False

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

    def to_request(self) -> ProcessingRequest:
        """Snapshot active fields; never pass hidden stale values to processing."""
        def path(value: str) -> Path:
            return Path(value).expanduser().absolute()

        def chart(image: str, target: str | None, cie: str) -> ChartInput:
            return ChartInput(path(image), target or "", path(cie) if cie.strip() else None)

        if not self.output_directory.strip():
            raise ValueError("Select a results folder.")
        return ProcessingRequest(
            output_directory=path(self.output_directory),
            profiling=chart(self.profiling_image, self.profiling_target, self.profiling_cie)
            if self.is_creation else None,
            evaluation=chart(self.effective_evaluation_image, self.evaluation_target, self.evaluation_cie)
            if not self.is_creation or self.has_additional else None,
            existing_icc=path(self.existing_icc) if not self.is_creation and self.existing_icc else None,
            profile_algorithm=self.profile_algorithm if self.is_creation else DEFAULT_PROFILE_ALGORITHM,
            profile_complexity=self.profile_complexity if self.is_creation else DEFAULT_PROFILE_COMPLEXITY,
        )

    @property
    def step_label(self) -> str:
        steps = ["welcome", "inputs", "additional", "save", "review"] if self.is_creation else [
            "welcome", "inputs", "save", "review"
        ]
        titles = {
            "welcome": "Choose a workflow",
            "inputs": "Profiling inputs" if self.is_creation else "Evaluation inputs",
            "additional": "Evaluation",
            "save": "Save results",
            "review": "Review",
        }
        return f"Step {steps.index(self.step) + 1} of {len(steps)} · {titles[self.step]}"

    @property
    def expected_outputs(self) -> str:
        if not self.is_creation:
            return "Evaluation report"
        if self.has_additional:
            return "ICC profile, creation report, and evaluation report"
        return "ICC profile and creation report"

    def review_rows(self) -> list[dict[str, str]]:
        selections = [("Workflow", WORKFLOWS[self.workflow])]
        if self.is_creation:
            selections.extend([
                ("Profiling TIFF", self.profiling_image),
                ("Profiling color target type", self.profiling_target),
                ("Profiling target CIE reference values", self.profiling_cie or "Chart generic reference values"),
                ("Profile algorithm", PROFILE_ALGORITHM_LABELS[self.profile_algorithm]),
                ("Profile complexity", PROFILE_COMPLEXITY_LABELS[self.profile_complexity]),
                ("Evaluation", ADDITIONAL_EVALUATION[self.additional]),
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
            ("Results folder", self.output_directory),
            ("Expected outputs", self.expected_outputs),
        ])
        return [{"setting": setting, "selection": str(value)} for setting, value in selections]


def _target_options() -> list[str]:
    return supported_targets()


def _field_help(label: str, explanation: str, marker: str) -> ui.button:
    """Explain a field on hover/focus, with a readable dialog for click or tap."""
    def show_help() -> None:
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-md gap-3"):
            ui.label(label).classes("text-lg font-medium")
            ui.label(explanation).classes("text-sm text-slate-600 leading-relaxed")
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.on_value_change(lambda event: dialog.delete() if not event.value else None)
        dialog.open()

    button = ui.button(icon="info_outline", on_click=show_help).props(
        f'flat round dense aria-label="About {label}"'
    ).classes("shrink-0 text-slate-600").mark(marker)
    with button:
        tooltip = ui.tooltip(explanation).classes("max-w-xs").style("font-size: 14px; line-height: 1.5")
    button.on("focusin", lambda: tooltip.run_method("show"))
    button.on("focusout", lambda: tooltip.run_method("hide"))
    return button


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
        )
        explanations = {
            "profiling_image": "TIFF image containing the color target used to create the ICC profile.",
            "evaluation_image": "TIFF image containing the color target used to evaluate the ICC profile.",
            "profiling_cie": "Optional .cie file with measurements for your specific profiling target. Leave blank to use generic reference values for the selected target type.",
            "evaluation_cie": "Optional .cie file with measurements for your specific evaluation target. Leave blank to use generic reference values for the selected target type.",
            "existing_icc": "Select the existing ICC profile whose color accuracy you want to evaluate.",
            "output_directory": "Folder where Arte-Profiler will save generated profiles, PDF reports, the processing log, and supporting files. Browse to select a folder or create a new one.",
        }
        _field_help(label, explanations[attribute], f"help-{attribute}")
    return row


def _target_fields(
    prefix: str, state: _FormState, attribute: str, targets: list[str], *,
    require_different: bool = False,
) -> ui.select:
    label = f"{prefix} color target type"
    with ui.row().classes("w-full flex-nowrap items-center gap-2"):
        selector = ui.select(targets, label=label).props("outlined").classes(
            "flex-1 min-w-0"
        ).bind_value(state, f"{attribute}_target")
        explanation = f"Select the type of physical color target present in the {prefix.lower()} TIFF image."
        if require_different:
            explanation += " The evaluation target type must be different from the profiling target type."
        _field_help(label, explanation, f"help-{attribute}_target")
    with ui.expansion("Advanced options", icon="tune").classes("w-full").bind_value(
        state, f"{attribute}_advanced"
    ) as advanced:
        with ui.column().classes("w-full gap-3").bind_visibility_from(advanced, "value"):
            _path_field(
                f"{prefix} target CIE reference values (optional)", state, f"{attribute}_cie", suffixes=(".cie",),
            )
    return selector


def _chart_error(prefix: str, image: str, target: str | None, cie: str, targets: list[str]) -> str:
    error = file_error(image, f"{prefix} TIFF image", (".tif", ".tiff"), tiff=True)
    if error:
        return error
    if target not in targets:
        return f"Select the {prefix.lower()} color target type."
    return file_error(cie, f"{prefix} CIE reference", (".cie",), optional=True)


def _existing_inputs_error(icc: str, image: str, target: str | None, cie: str, targets: list[str]) -> str:
    return file_error(icc, "Existing ICC profile", (".icc", ".icm")) or _chart_error(
        "Evaluation", image, target, cie, targets,
    )


def create_page() -> None:
    """Build a wizard with persistent values for each browser session."""
    state = _FormState()
    targets = _target_options()
    controller = JobController()
    card_classes = "w-full gap-4 p-6 shadow-sm border border-slate-200"

    def go_to(step: str) -> None:
        state.input_error = ""
        state.additional_error = ""
        state.save_error = ""
        if step == "review":
            review.rows = state.review_rows()
        state.step = step

    async def check_step(check, error_attribute: str, destination: str) -> None:
        if state.checking_files:
            return
        setattr(state, error_attribute, "")
        snapshot = vars(state).copy()
        state.checking_files = True
        try:
            error = await run.io_bound(check)
            current = vars(state).copy()
            current["checking_files"] = False
            # Ignore an old result if the user edited a field or navigated back.
            if current != snapshot:
                return
            if error:
                setattr(state, error_attribute, error)
            else:
                go_to(destination)
        finally:
            state.checking_files = False

    async def continue_inputs() -> None:
        sync_evaluation_targets()
        if state.is_creation:
            check = partial(_chart_error,
                "Profiling", state.profiling_image, state.profiling_target, state.profiling_cie, targets,
            )
        else:
            check = partial(_existing_inputs_error,
                state.existing_icc, state.evaluation_image, state.evaluation_target, state.evaluation_cie, targets,
            )
        await check_step(check, "input_error", "additional" if state.is_creation else "save")

    async def review_creation() -> None:
        if state.has_additional and state.evaluation_target == state.profiling_target:
            state.additional_error = "The evaluation target type must differ from the profiling target type."
            return
        if not state.has_additional:
            go_to("save")
            return
        check = partial(_chart_error,
            "Evaluation", state.effective_evaluation_image, state.evaluation_target, state.evaluation_cie, targets,
        )
        await check_step(check, "additional_error", "save")

    def review_results() -> None:
        if not state.output_directory.strip():
            state.save_error = "Select a results folder."
        else:
            go_to("review")

    def sync_evaluation_targets() -> None:
        options = [target for target in targets if not state.is_creation or target != state.profiling_target]
        if state.is_creation and state.evaluation_target == state.profiling_target:
            state.evaluation_target = None
            state.evaluation_cie = ""
        for selector in (existing_target, additional_target):
            selector.set_options(options, value=state.evaluation_target)

    def show_artifact(label: str, path: Path) -> None:
        with results:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(label).classes("font-medium")
                ui.label(str(path)).classes("text-sm text-slate-600 break-all")
                if path.is_file():
                    ui.button("Download", on_click=lambda: ui.download(path)).props("flat dense")

    async def collect_events(request: ProcessingRequest) -> None:
        # This task outlives a disconnected page so the worker is always reaped.
        try:
            while controller.running:
                for event in controller.poll():
                    if run_button.is_deleted:
                        continue
                    if event.kind == "log":
                        processing_log.push(event.message)
                    elif event.kind == "artifact":
                        show_artifact(event.message, event.path)
                        if request.profiling is not None and event.message == "ICC profile":
                            with results:
                                ui.label(f"Profile algorithm: {PROFILE_ALGORITHM_LABELS[request.profile_algorithm]}").classes(
                                    "text-sm text-slate-600"
                                )
                                ui.label(f"Profile complexity: {PROFILE_COMPLEXITY_LABELS[request.profile_complexity]}").classes(
                                    "text-sm text-slate-600"
                                )
                    else:
                        status.set_text(event.message)
                        if event.kind == "stage":
                            processing_log.push(event.message)
                        elif event.kind == "failed":
                            status.classes(add="text-red-800", remove="text-slate-600")
                            processing_log.push(event.message)
                await asyncio.sleep(0.1)
        finally:
            if not run_button.is_deleted:
                log_path = request.output_directory / "profiling.log"
                if log_path.is_file():
                    show_artifact("Processing log", log_path)
                run_button.enable()
                back_button.enable()
                spinner.set_visibility(False)

    def start_processing() -> None:
        try:
            request = state.to_request()
            protect_bundle_output(request.output_directory)
            controller.start(request)
        except (OSError, ValueError, RuntimeError) as error:
            status.set_text(str(error))
            status.classes(add="text-red-800", remove="text-slate-600")
            return
        results.clear()
        processing_log.clear()
        status.classes(add="text-slate-600", remove="text-red-800")
        status.set_text("Starting processing…")
        run_button.disable()
        back_button.disable()
        spinner.set_visibility(True)
        background_tasks.create(collect_events(request))

    ui.colors(primary="#245c66")
    with ui.column().classes("w-full max-w-4xl mx-auto gap-5 p-4 sm:p-8"):
        with ui.column().classes("gap-1"):
            ui.label("Arte-Profiler").classes("text-3xl font-semibold text-slate-900")
            ui.label("Camera profiling and color accuracy evaluation").classes("text-base text-slate-600")
        ui.label().bind_text_from(state, "step_label").classes("text-sm font-medium text-slate-600")

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="welcome"):
            ui.label("Choose a workflow").classes("text-xl font-medium")
            workflow_radio = ui.radio(
                WORKFLOWS, value=state.workflow, on_change=lambda: sync_evaluation_targets(),
            ).bind_value(state, "workflow").classes("w-full")
            explanations = {
                "create": "Generate and evaluate an ICC color profile using supported targets.",
                "existing": "Evaluate an existing ICC profile using a supported target.",
            }
            for index, (workflow, label) in enumerate(WORKFLOWS.items()):
                with workflow_radio.add_slot(f"label-{index}"):
                    with ui.row().classes("items-center flex-nowrap gap-2"):
                        ui.label(label)
                        _field_help(label, explanations[workflow], f"help-workflow-{workflow}").on(
                            "click", js_handler="event => event.stopPropagation()",
                        )
            ui.label("Supported targets").classes("font-medium")
            ui.label(", ".join(targets)).classes("text-sm text-slate-600")
            ui.button("Continue", on_click=lambda: go_to("inputs"))

        with ui.column().classes("w-full gap-2 text-xs text-slate-600").bind_visibility_from(
            state, "step", value="welcome",
        ):
            link_classes = (
                "text-xs text-primary underline-offset-2 hover:opacity-75 "
                "focus-visible:outline focus-visible:outline-offset-2"
            )
            ui.separator()
            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("article", size="16px")
                    ui.label("Related publication").classes("text-xs font-medium")
                with ui.element("div").classes("text-xs leading-relaxed"):
                    ui.label(
                        'Alessandra Marrocchesi, Robert G. Erdmann, and Douglas Peterson. '
                        '“Arte-Profiler: An Open-Source Tool for Color Profiling and Accuracy '
                        'Assessment in Cultural Heritage Photography”. In: Cultural Heritage '
                        'Analysis for New Generations. Edited by Christian Degrigny, Sony George, '
                        'and Jon Y. Hardeberg. Routledge, 2026, pp. 139–154. '
                    ).classes("inline")
                    ui.link(
                        "doi: 10.4324/9781003441366-11", "https://doi.org/10.4324/9781003441366-11", new_tab=True,
                    ).classes(link_classes)
            with ui.row().classes("mt-2 items-center gap-x-5 gap-y-2"):
                with ui.row().classes("items-center gap-1"):
                    ui.html(_GITHUB_MARK_SVG, tag="span").classes("inline-flex shrink-0")
                    ui.link("GitHub repository", "https://github.com/amarrocc/arte-profiler", new_tab=True).classes(link_classes)
                with ui.row().classes("items-center gap-1"):
                    ui.icon("description", size="16px")
                    ui.link(
                        "GPL v3 license", "https://github.com/amarrocc/arte-profiler/blob/main/LICENSE", new_tab=True,
                    ).classes(link_classes)

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="inputs"):
            with ui.column().classes("w-full gap-4").bind_visibility_from(state, "is_creation"):
                ui.label("Profiling inputs").classes("text-xl font-medium")
                _path_field("Profiling TIFF", state, "profiling_image", suffixes=(".tif", ".tiff"))
                profiling_target = _target_fields("Profiling", state, "profiling", targets)
                profiling_target.on_value_change(lambda: sync_evaluation_targets())
                with ui.expansion("Advanced profile settings", icon="tune").classes("w-full").bind_value(
                    state, "profile_settings_advanced",
                ) as profile_settings:
                    with ui.column().classes("w-full gap-3").bind_visibility_from(profile_settings, "value"):
                        ui.select(
                            PROFILE_ALGORITHM_LABELS, label="Profile algorithm (colprof -a)",
                        ).props("outlined").classes("w-full").bind_value(state, "profile_algorithm").mark("profile-algorithm")
                        with ui.row().classes("w-full flex-nowrap items-center gap-2"):
                            ui.select(
                                PROFILE_COMPLEXITY_LABELS, label="Profile complexity (colprof -q)",
                            ).props("outlined").classes("flex-1 min-w-0").bind_value(state, "profile_complexity").mark("profile-complexity")
                            _field_help(
                                "Profile complexity",
                                "Controls fitting effort or lookup-table resolution. Higher complexity does not guarantee better color accuracy.",
                                "help-profile_complexity",
                            )
            with ui.column().classes("w-full gap-4").bind_visibility_from(
                state, "is_creation", backward=lambda value: not value
            ):
                ui.label("Evaluation inputs").classes("text-xl font-medium")
                _path_field("Existing ICC profile", state, "existing_icc", suffixes=(".icc", ".icm"))
                _path_field("Evaluation TIFF", state, "evaluation_image", suffixes=(".tif", ".tiff"))
                existing_target = _target_fields("Evaluation", state, "evaluation", targets)
            ui.label().bind_text_from(state, "input_error").classes("text-sm text-red-800")
            ui.label("Checking selected files…").classes("text-sm text-slate-600").bind_visibility_from(state, "checking_files")
            with ui.row().classes("w-full justify-between"):
                ui.button("Back", on_click=lambda: go_to("welcome")).props("outline")
                ui.button("Continue", on_click=continue_inputs).bind_enabled_from(
                    state, "checking_files", backward=lambda value: not value,
                )

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
                    state, "profiling_image", backward=lambda path: f"Evaluation TIFF: {path}"
                ).classes("text-sm text-slate-600 break-all").bind_visibility_from(state, "reuses_profiling_image")
                _path_field(
                    "Evaluation TIFF", state, "evaluation_image", suffixes=(".tif", ".tiff"),
                ).bind_visibility_from(state, "needs_separate_image")
                additional_target = _target_fields("Evaluation", state, "evaluation", targets, require_different=True)
            ui.label().bind_text_from(state, "additional_error").classes("text-sm text-red-800")
            ui.label("Checking selected files…").classes("text-sm text-slate-600").bind_visibility_from(state, "checking_files")
            with ui.row().classes("w-full justify-between"):
                ui.button("Back", on_click=lambda: go_to("inputs")).props("outline")
                ui.button("Continue", on_click=review_creation).bind_enabled_from(
                    state, "checking_files", backward=lambda value: not value,
                )

        with ui.card().classes(card_classes).bind_visibility_from(state, "step", value="save"):
            ui.label("Save results").classes("text-xl font-medium")
            _path_field("Results folder", state, "output_directory", directory=True)
            ui.label().bind_text_from(state, "save_error").classes("text-sm text-red-800")
            with ui.row().classes("w-full justify-between"):
                ui.button(
                    "Back", on_click=lambda: go_to("additional" if state.is_creation else "inputs"),
                ).props("outline")
                ui.button("Review", on_click=review_results)

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
            with ui.row().classes("w-full items-center justify-between gap-3"):
                back_button = ui.button(
                    "Back", on_click=lambda: go_to("save")
                ).props("outline")
                run_button = ui.button("Run", icon="play_arrow", on_click=start_processing)
                spinner = ui.spinner(size="sm")
                spinner.set_visibility(False)
            ui.label("Existing output files with the same names will be replaced.").classes("text-sm text-slate-600")
            status = ui.label("Ready to process.").classes("text-sm text-slate-600").mark("processing-status")
            ui.separator()
            ui.label("Processing log").classes("text-lg font-medium")
            processing_log = ui.log(max_lines=500).classes("w-full h-48 border border-slate-200 rounded")
            ui.label("Results").classes("text-lg font-medium")
            results = ui.column().classes("w-full gap-2").mark("processing-results")
            with results:
                ui.label("No results yet.").classes("text-sm text-slate-600")
    sync_evaluation_targets()
