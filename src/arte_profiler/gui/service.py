"""GUI orchestration of the existing scientific API, without UI dependencies."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable

import yaml

from arte_profiler.profiling import ProfileCreator, ProfileEvaluator


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

    def validate(self) -> None:
        if (self.profiling is None) == (self.existing_icc is None):
            raise ValueError("Select either profile creation or an existing ICC profile.")
        if self.profiling is None and self.evaluation is None:
            raise ValueError("Select an evaluation target and TIFF.")
        if self.profiling and self.evaluation and self.profiling.target == self.evaluation.target:
            raise ValueError("The evaluation target type must differ from the profiling target type.")
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
