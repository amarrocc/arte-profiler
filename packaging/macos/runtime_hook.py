"""Run in the frozen server AND workers, before scientific or GUI imports."""

from arte_profiler.gui.packaged import prepare_runtime

prepare_runtime()
