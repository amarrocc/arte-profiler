"""GUI orchestration of the existing scientific API, without UI dependencies."""

from dataclasses import dataclass
from importlib.resources import files
from inspect import signature
from pathlib import Path
from typing import Callable

import yaml

from arte_profiler.profiling import (
    PROFILE_ALGORITHMS, PROFILE_COMPLEXITIES, ProfileCreator, ProfileEvaluator,
    _validate_profile_options,
)

_creator_parameters = signature(ProfileCreator).parameters
DEFAULT_PROFILE_ALGORITHM = _creator_parameters["profile_algorithm"].default
DEFAULT_PROFILE_COMPLEXITY = _creator_parameters["profile_complexity"].default
PROFILE_ALGORITHM_LABELS = {
    code: f"{code} — " + {
        "g": "Gamma + matrix", "s": "Shaper + matrix", "m": "Matrix (linear curves)",
        "l": "Lab LUT", "x": "XYZ LUT",
    }.get(code, code) + (" (default)" if code == DEFAULT_PROFILE_ALGORITHM else "")
    for code in PROFILE_ALGORITHMS
}
PROFILE_COMPLEXITY_LABELS = {
    code: f"{code} — " + {"l": "Low", "m": "Medium", "h": "High"}.get(code, code)
    + (" (default)" if code == DEFAULT_PROFILE_COMPLEXITY else "")
    for code in PROFILE_COMPLEXITIES
}


def supported_targets() -> list[str]:
    manifest = files("arte_profiler") / "data" / "targets" / "targets_manifest.yaml"
    with manifest.open(encoding="utf-8") as stream:
        return list(yaml.safe_load(stream))


@dataclass(frozen=True)
class ChartInput:
    image: Path
    target: str
    cie: Path | None = None


@dataclass(frozen=True)
class ProcessingRequest:
    output_directory: Path
    profiling: ChartInput | None = None
    evaluation: ChartInput | None = None
    existing_icc: Path | None = None
    profile_algorithm: str = DEFAULT_PROFILE_ALGORITHM
    profile_complexity: str = DEFAULT_PROFILE_COMPLEXITY

    def validate(self) -> None:
        if (self.profiling is None) == (self.existing_icc is None):
            raise ValueError("Select either profile creation or an existing ICC profile.")
        if self.profiling is None and self.evaluation is None:
            raise ValueError("Select an evaluation target and TIFF.")
        if self.profiling and self.evaluation and self.profiling.target == self.evaluation.target:
            raise ValueError("The evaluation target type must differ from the profiling target type.")
        if self.profiling is not None:
            _validate_profile_options(self.profile_algorithm, self.profile_complexity)
        targets = supported_targets()
        for chart in (self.profiling, self.evaluation):
            if chart is None:
                continue
            if chart.target not in targets:
                raise ValueError(f"Unsupported target type: {chart.target}")
            if chart.image.suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError("Select a TIFF image (.tif or .tiff).")
            for path in (chart.image, chart.cie):
                if path is not None and not path.is_file():
                    raise FileNotFoundError(f"File {path} not found.")
        if self.existing_icc is not None and not self.existing_icc.is_file():
            raise FileNotFoundError(f"File {self.existing_icc} not found.")
        if self.output_directory.exists() and not self.output_directory.is_dir():
            raise ValueError("The output path must be a directory.")


@dataclass(frozen=True)
class JobEvent:
    kind: str
    message: str
    path: Path | None = None


def process_request(
    request: ProcessingRequest,
    *,
    logger_name: str,
    emit: Callable[[JobEvent], None],
) -> tuple[JobEvent, ...]:
    """Match CLI stage order, defaults, and measurement reuse; propagate failures."""
    request.validate()
    common = {"folder": request.output_directory, "logger_name": logger_name}
    artifacts: list[JobEvent] = []

    def completed(label: str, path: Path) -> None:
        artifact = JobEvent("artifact", label, path)
        artifacts.append(artifact)
        emit(artifact)

    if request.profiling is not None:
        chart = request.profiling
        emit(JobEvent("stage", "Building ICC profile…"))
        creator = ProfileCreator(
            chart_tif=chart.image, chart_type=chart.target, chart_cie=chart.cie, **common,
            profile_algorithm=request.profile_algorithm, profile_complexity=request.profile_complexity,
        )
        icc = creator.build_profile()
        completed("ICC profile", icc)

        emit(JobEvent("stage", "Generating creation report…"))
        evaluator = ProfileEvaluator(
            chart_tif=chart.image, chart_type=chart.target, chart_cie=chart.cie,
            in_icc=icc, patch_data=creator.df, **common,
        )
        report = evaluator.evaluate_profile(
            report_title="Profile Creation Report", report_filename="profile_creation_report.pdf",
        )
        completed("Creation report", report)
    else:
        icc = request.existing_icc

    if request.evaluation is not None:
        chart = request.evaluation
        emit(JobEvent("stage", "Evaluating ICC profile…"))
        evaluator = ProfileEvaluator(
            chart_tif=chart.image, chart_type=chart.target, chart_cie=chart.cie,
            in_icc=icc, patch_data=None, **common,
        )
        report = evaluator.evaluate_profile(
            report_title="Profile Evaluation Report", report_filename="profile_evaluation_report.pdf",
        )
        completed("Evaluation report", report)

    return tuple(artifacts)
