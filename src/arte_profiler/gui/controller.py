"""Run one scientific job at a time in a spawned process and relay its events."""

import logging
import multiprocessing
from queue import Empty
from threading import Lock
from uuid import uuid4

from .service import JobEvent, ProcessingRequest, process_request

_job_lock = Lock()


def _worker(request: ProcessingRequest, events, logger_name: str) -> None:
    class EventHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.put(JobEvent("log", self.format(record)))

    handler = EventHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    loggers = [logging.getLogger(logger_name), logging.getLogger(f"{logger_name}.command")]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    try:
        # Keep plotting independent of native GUI toolkits, including on macOS.
        import matplotlib
        matplotlib.use("Agg")
        process_request(request, logger_name=logger_name, emit=events.put)
    except Exception as error:
        loggers[0].exception("Processing failed")
        events.put(JobEvent("failed", f"{type(error).__name__}: {error}"))
    else:
        events.put(JobEvent("done", "Processing completed."))
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
        handler.close()
        events.close()
        events.join_thread()


class JobController:
    """GUI-free controller; poll from the UI, never update widgets from the worker."""

    def __init__(self) -> None:
        self._process = None
        self._events = None
        self._terminal_received = False

    @property
    def running(self) -> bool:
        return self._process is not None

    def start(self, request: ProcessingRequest) -> None:
        request.validate()
        if self.running or not _job_lock.acquire(blocking=False):
            raise RuntimeError("Another processing job is running. Wait for it to finish.")
        try:
            context = multiprocessing.get_context("spawn")
            self._events = context.Queue()
            self._process = context.Process(
                target=_worker, args=(request, self._events, f"arte_profiler.gui.{uuid4().hex}"),
            )
            self._terminal_received = False
            self._process.start()
        except Exception:
            self._process = None
            if self._events is not None:
                self._events.close()
                self._events = None
            _job_lock.release()
            raise

    def poll(self) -> list[JobEvent]:
        if self._process is None:
            return []
        result = []
        # Bound each UI tick so a verbose subprocess cannot monopolize the event loop.
        for _ in range(200):
            try:
                event = self._events.get_nowait()
            except Empty:
                break
            result.append(event)
            if event.kind in {"done", "failed"}:
                self._terminal_received = True
        else:
            return result
        if not self._process.is_alive():
            if not self._terminal_received:
                result.append(JobEvent("failed", f"Processing worker exited (code {self._process.exitcode})."))
            self._process.join()
            self._process.close()
            self._process = None
            self._events.close()
            self._events = None
            _job_lock.release()
        return result
