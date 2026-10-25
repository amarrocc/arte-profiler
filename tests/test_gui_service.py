"""Scientific orchestration contracts: compare the GUI service with unmodified main()."""

from argparse import Namespace
from pathlib import Path

import pytest

from arte_profiler import profiling
from arte_profiler.gui import service
from arte_profiler.gui.service import ChartInput, ProcessingRequest


def make_request(tmp_path, mode, custom=False):
    image = tmp_path / "profiling.tiff"
    image.write_text("TIFF placeholder for API spies")
    other = tmp_path / "evaluation.tiff"
    other.write_text("TIFF placeholder for API spies")
    cie = tmp_path / "measured.cie"
    cie.write_text("CIE placeholder for API spies")
    icc = tmp_path / "existing.icc"
    icc.write_text("ICC placeholder for API spies")
    build = ChartInput(image, "ColorCheckerSG", cie if custom else None)
    evaluation = ChartInput(image if mode == "same" else other, "DT-NGT2", cie if custom else None)
    return ProcessingRequest(
        tmp_path / "results", profiling=None if mode == "existing" else build,
        evaluation=None if mode == "none" else evaluation,
        existing_icc=icc if mode == "existing" else None,
    )


@pytest.mark.parametrize("mode", ["none", "same", "separate", "existing"])
@pytest.mark.parametrize("custom", [False, True])
def test_service_matches_cli_calls_and_defaults(tmp_path, monkeypatch, mode, custom):
    request = make_request(tmp_path, mode, custom)
    calls = []
    measurements = object()

    class Creator:
        def __init__(self, chart_tif, chart_type, chart_cie=None, folder=None, logger_name=None):
            calls.append(("creator", chart_tif, chart_type, chart_cie, folder, logger_name))
            self.folder = folder
            self.df = measurements

        def build_profile(self, fiducial=None, profile_name="input_profile.icc"):
            calls.append(("build", fiducial, profile_name))
            self.in_icc = self.folder / profile_name
            return self.in_icc

    class Evaluator:
        def __init__(self, chart_tif, chart_type, in_icc, chart_cie=None, folder=None,
                     patch_data=None, logger_name=None):
            calls.append(("evaluator", chart_tif, chart_type, in_icc, chart_cie, folder, patch_data, logger_name))
            self.folder = folder

        def evaluate_profile(self, fiducial=None, report_title="Profiling Report",
                             report_filename="profiling_report.pdf"):
            calls.append(("evaluate", fiducial, report_title, report_filename))
            return self.folder / report_filename

    for module in (profiling, service):
        monkeypatch.setattr(module, "ProfileCreator", Creator)
        monkeypatch.setattr(module, "ProfileEvaluator", Evaluator)
    build = request.profiling
    # The CLI requires test args even when only the creation check is wanted.
    test = request.evaluation or build
    monkeypatch.setattr(profiling, "parse_args", lambda: Namespace(
        build_tif=build.image if build else None, build_type=build.target if build else None,
        build_cie=build.cie if build else None, test_tif=test.image, test_type=test.target,
        test_cie=test.cie, in_icc=request.existing_icc, out_folder=request.output_directory,
        fiducial_build=None, fiducial_test=None,
    ))
    profiling.main()
    cli_calls = calls.copy()
    calls.clear()
    events = []
    artifacts = service.process_request(request, logger_name="profiling", emit=events.append)
    assert calls == cli_calls
    evaluator_calls = [call for call in calls if call[0] == "evaluator"]
    if build:
        assert evaluator_calls[0][-2] is measurements
    if request.evaluation:
        assert evaluator_calls[-1][-2] is None
        assert calls[-1] == ("evaluate", None, "Profile Evaluation Report", "profile_evaluation_report.pdf")
    assert [event for event in events if event.kind == "artifact"] == list(artifacts)
    assert len(artifacts) == (1 if mode == "existing" else 2 if mode == "none" else 3)


def test_service_rejects_same_target_before_scientific_calls(tmp_path, monkeypatch):
    request = make_request(tmp_path, "same")
    request = ProcessingRequest(request.output_directory, request.profiling, request.profiling)
    monkeypatch.setattr(service, "ProfileCreator", lambda **kwargs: pytest.fail("Backend must not be called"))
    with pytest.raises(ValueError, match="must differ"):
        service.process_request(request, logger_name="test", emit=lambda _: None)
    assert not request.output_directory.exists()


def test_creation_report_failure_preserves_icc_event_and_stops_evaluation(tmp_path, monkeypatch):
    request = make_request(tmp_path, "separate")
    measurements = object()
    evaluators = []

    class Creator:
        def __init__(self, **kwargs):
            self.df = measurements

        def build_profile(self):
            return request.output_directory / "input_profile.icc"

    class Evaluator:
        def __init__(self, **kwargs):
            evaluators.append(kwargs)

        def evaluate_profile(self, **kwargs):
            raise RuntimeError("Report failed")

    monkeypatch.setattr(service, "ProfileCreator", Creator)
    monkeypatch.setattr(service, "ProfileEvaluator", Evaluator)
    events = []
    with pytest.raises(RuntimeError, match="Report failed"):
        service.process_request(request, logger_name="test", emit=events.append)
    assert len(evaluators) == 1
    assert evaluators[0]["patch_data"] is measurements
    assert [event.message for event in events if event.kind == "artifact"] == ["ICC profile"]


def test_missing_input_does_not_create_output(tmp_path):
    request = ProcessingRequest(tmp_path / "results", profiling=ChartInput(tmp_path / "missing.tiff", "ColorCheckerSG"))
    with pytest.raises(FileNotFoundError):
        request.validate()
    assert not request.output_directory.exists()
