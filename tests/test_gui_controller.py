"""Worker lifecycle and log forwarding, independent of browser widgets."""

import logging
import os
from pathlib import Path
from queue import Queue
import time
from uuid import uuid4

import pytest

from arte_profiler.gui import controller
from arte_profiler.gui.service import ChartInput, JobEvent, ProcessingRequest


class EventQueue(Queue):
    def close(self):
        pass

    def join_thread(self):
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_worker_forwards_both_loggers_and_terminal_event(tmp_path, monkeypatch, fail):
    name = f"test.gui.{uuid4().hex}"
    events = EventQueue()

    def process(request, *, logger_name, emit):
        logger = logging.getLogger(logger_name)
        command = logging.getLogger(f"{logger_name}.command")
        command.propagate = False  # As configured by the existing backend.
        logger.warning("Patch warning")
        command.info("Argyll output")
        emit(JobEvent("artifact", "ICC profile", tmp_path / "input_profile.icc"))
        if fail:
            raise RuntimeError("scanin failed")

    monkeypatch.setattr(controller, "process_request", process)
    controller._worker(ProcessingRequest(tmp_path), events, name)
    received = list(events.queue)
    messages = [event.message for event in received if event.kind == "log"]
    assert sum("Patch warning" in message for message in messages) == 1
    assert sum("Argyll output" in message for message in messages) == 1
    assert received[-1].kind == ("failed" if fail else "done")
    if fail:
        assert "RuntimeError: scanin failed" in received[-1].message
        assert any("Traceback" in message for message in messages)
    assert not logging.getLogger(name).handlers
    assert not logging.getLogger(f"{name}.command").handlers


def collect(job, timeout=60):
    deadline = time.monotonic() + timeout
    received = []
    while job.running:
        received.extend(job.poll())
        if time.monotonic() > deadline:
            # Test cleanup only; the application does not expose unsupported cancellation.
            job._process.terminate()
            job._process.join()
            job.poll()
            pytest.fail("Worker did not finish")
        time.sleep(0.05)
    return received


def test_spawned_worker_failure_releases_job_lock(tmp_path):
    image = tmp_path / "invalid.tiff"
    image.write_text("Not a TIFF")
    request = ProcessingRequest(tmp_path / "results", profiling=ChartInput(image, "ColorCheckerSG"))
    first, second = controller.JobController(), controller.JobController()
    first.start(request)
    with pytest.raises(RuntimeError, match="Another processing job"):
        second.start(request)
    received = collect(first)
    assert received[-1].kind == "failed"
    assert "Fiducial auto-recognition failed" in received[-1].message
    assert any(event.kind == "log" and "Fiducial detection failed" in event.message for event in received)
    second.start(request)
    assert collect(second)[-1].kind == "failed"


@pytest.mark.skipif(os.environ.get("ARTE_PROFILER_SCIENCE_SMOKE") != "1", reason="Opt-in real ArgyllCMS execution")
def test_real_worker_creation_additional_evaluation_and_existing_profile(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    # Same-TIFF mode exercises fresh extraction for each different target type.
    request = ProcessingRequest(
        tmp_path / "created",
        profiling=ChartInput(examples / "sample_dual_capture.tiff", "ColorCheckerSG"),
        evaluation=ChartInput(examples / "sample_dual_capture.tiff", "DT-NGT2"),
    )
    job = controller.JobController()
    job.start(request)
    received = collect(job, timeout=180)
    assert received[-1].kind == "done", received[-1].message
    artifacts = [event for event in received if event.kind == "artifact"]
    assert [event.message for event in artifacts] == ["ICC profile", "Creation report", "Evaluation report"]
    for artifact in artifacts:
        assert artifact.path.stat().st_size > 0
    assert artifacts[1].path.read_bytes().startswith(b"%PDF")
    assert artifacts[2].path.read_bytes().startswith(b"%PDF")
    assert (request.output_directory / "profiling.log").is_file()
    # Evaluate the generated ICC against a separate TIFF, without rebuilding it.
    job.start(ProcessingRequest(
        tmp_path / "evaluated", existing_icc=artifacts[0].path,
        evaluation=ChartInput(examples / "sample_DT-NGT2.tiff", "DT-NGT2"),
    ))
    received = collect(job, timeout=180)
    assert received[-1].kind == "done", received[-1].message
    assert [event.message for event in received if event.kind == "artifact"] == ["Evaluation report"]
    assert not (tmp_path / "evaluated" / "input_profile.icc").exists()
