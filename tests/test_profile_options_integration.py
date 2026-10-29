"""Opt-in, real bundled-Argyll checks on a supplied, known RGB INPUT TI3 file.

Set ARTE_PROFILER_TEST_TI3 to enable. Optional ARTE_PROFILER_TEST_OUTPUT_ROOT
retains evidence in a NEW directory; otherwise TemporaryDirectory cleans up.
No image extraction or accuracy claims are made by these tests.
"""

from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from arte_profiler import profiling


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def icc_tags(path):
    """Read raw tag payloads for evidence comparison, without transforming colors."""
    data = path.read_bytes()
    if data[36:40] != b"acsp" or len(data) != struct.unpack_from(">I", data)[0]:
        raise AssertionError(f"Invalid ICC profile: {path}")
    tags = {}
    for index in range(struct.unpack_from(">I", data, 128)[0]):
        name, offset, size = struct.unpack_from(">4sII", data, 132 + index * 12)
        tags[name.decode("ascii")] = data[offset:offset + size]
    return data, tags


@unittest.skipUnless(os.environ.get("ARTE_PROFILER_TEST_TI3"), "Set ARTE_PROFILER_TEST_TI3 for real Argyll tests")
class BundledArgyllProfileOptionsTests(unittest.TestCase):
    def test_all_configurations_and_default_against_original_command(self):
        source = Path(os.environ["ARTE_PROFILER_TEST_TI3"]).expanduser().resolve()
        self.assertTrue(source.is_file(), source)
        source_hash = digest(source)
        retained = os.environ.get("ARTE_PROFILER_TEST_OUTPUT_ROOT")
        if retained:
            root = Path(retained).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=False)  # Never replace previous results.
        else:
            temporary = tempfile.TemporaryDirectory(prefix="arte-profile-options-")
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        print(f"\nReal Argyll test evidence: {root}", flush=True)
        image = root / "constructor-only.tiff"
        image.touch()  # No TIFF processing is exercised by icc_from_ti3().
        ti3 = root / "measurements.ti3"
        shutil.copy2(source, ti3)
        self.assertIn('DEVICE_CLASS "INPUT"', ti3.read_text())
        frame = profiling.profiling_utils.ti3_to_dataframe(ti3)
        rgb_text = frame.to_csv(sep="\t", columns=["RGB_R", "RGB_G", "RGB_B"], header=False, index=False)
        bins = Path(profiling.profiling_utils.get_argyll_bin_path()).resolve()
        tools = {name: (bins / (name + (".exe" if os.name == "nt" else ""))).resolve()
                 for name in ("colprof", "icclu")}
        package = Path(profiling.__file__).resolve().parent
        for executable in tools.values():
            self.assertTrue(executable.is_file(), executable)
            self.assertTrue(executable.is_relative_to(package / "tools"), executable)
        help_result = subprocess.run([str(tools["colprof"]), "-?"], capture_output=True, text=True, timeout=30)
        help_text = help_result.stdout + help_result.stderr
        self.assertIn("gamma+matrix", help_text)
        self.assertIn("Lab cLUT", help_text)
        self.assertIn("XYZ cLUT", help_text)
        self.assertIn("Quality", help_text)
        (root / "colprof_version.txt").write_text(help_text)
        lookup_help = subprocess.run([str(tools["icclu"]), "-?"], capture_output=True, text=True, timeout=30)
        (root / "icclu_version.txt").write_text(lookup_help.stdout + lookup_help.stderr)
        commands = []

        def execute(argv, label, input_text=None):
            started = time.monotonic()
            result = subprocess.run(argv, input=input_text, capture_output=True, text=True, timeout=300)
            (root / f"{label}.stdout.txt").write_text(result.stdout)
            (root / f"{label}.stderr.txt").write_text(result.stderr)
            record = {"argv": argv, "returncode": result.returncode,
                      "elapsed_seconds": time.monotonic() - started,
                      "stdin_sha256": hashlib.sha256(input_text.encode()).hexdigest() if input_text else None}
            commands.append(record)
            (root / f"{label}.command.json").write_text(json.dumps(record, indent=2))
            self.assertEqual(result.returncode, 0, f"{label}: see {root / (label + '.stderr.txt')}")
            return result

        # Hardcoded pre-change command, not constructed from production option constants.
        legacy_path = root / "legacy.icc"
        legacy = [str(tools["colprof"]), "-v", "-Z", "a", "-ua", "-a", "g", "-b", "n", "-q", "m",
                  "-O", str(legacy_path), str(ti3.with_suffix(""))]
        execute(legacy, "legacy_build")
        legacy_header, legacy_tags = icc_tags(legacy_path)
        numeric_legacy = execute([str(tools["icclu"]), "-s", "100", "-p", "l", "-v", "0", "-ia",
                                  str(legacy_path)], "legacy_lookup", rgb_text).stdout
        logger = logging.getLogger(f"profile-options-integration.{root.name}")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        results = []
        # Default is tested with BOTH options omitted. Every other pair is explicit.
        pairs = [("g", "m"), *[pair for pair in itertools.product("gsmlx", "lmh") if pair != ("g", "m")]]
        for algorithm, complexity in pairs:
            with self.subTest(algorithm=algorithm, complexity=complexity):
                label = f"a_{algorithm}_q_{complexity}"
                options = {} if (algorithm, complexity) == ("g", "m") else {
                    "profile_algorithm": algorithm, "profile_complexity": complexity}
                with patch.object(profiling.profiling_utils, "generate_logger", return_value=(logger, logger)):
                    creator = profiling.ProfileCreator(image, "ColorCheckerSG", folder=root, **options)
                creator.chart_ti3 = ti3

                def real_command(argv, command_logger):
                    # Real execution, with bounded duration and complete retained stdout/stderr.
                    return execute(argv, label + "_build").returncode

                print(f"Testing bundled colprof -a {algorithm} -q {complexity}", flush=True)
                with patch.object(profiling.profiling_utils, "run_command", side_effect=real_command):
                    profile = creator.icc_from_ti3(label + ".icc")
                header, tags = icc_tags(profile)
                self.assertEqual(header[12:16], b"scnr")
                self.assertEqual(header[16:20], b"RGB ")
                self.assertEqual(header[20:24], b"Lab " if algorithm == "l" else b"XYZ ")
                self.assertEqual(struct.unpack_from(">I", header, 64)[0], 3)
                self.assertEqual(tags["wtpt"], legacy_tags["wtpt"])
                self.assertFalse(any(tag.startswith("B2A") for tag in tags))
                if algorithm in "gsm":
                    for tag in ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"):
                        self.assertIn(tag, tags)
                else:
                    self.assertIn("A2B0", tags)
                lookup = execute([str(tools["icclu"]), "-s", "100", "-p", "l", "-v", "0", "-ia", str(profile)],
                                 label + "_lookup", rgb_text).stdout
                values = np.loadtxt(io.StringIO(lookup), ndmin=2)
                self.assertEqual(values.shape, (len(frame), 3))
                self.assertTrue(np.isfinite(values).all())
                if (algorithm, complexity) == ("g", "m"):
                    self.assertEqual(tags, legacy_tags)
                    # Whole-file differences from dates/IDs are not numerical differences.
                    self.assertEqual(header[:24], legacy_header[:24])
                    self.assertEqual(lookup, numeric_legacy)
                results.append({"algorithm": algorithm, "complexity": complexity,
                                "ICC_sha256": digest(profile), "lookup_patches": len(values)})
        self.assertEqual(len(results), 15)
        self.assertEqual(digest(source), source_hash)
        self.assertEqual(digest(ti3), source_hash)
        (root / "results.json").write_text(json.dumps({
            "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "source_ti3": str(source), "source_ti3_sha256": source_hash,
            "tools": {name: {"path": str(path), "sha256": digest(path)} for name, path in tools.items()},
            "default_tags_identical_to_original_command": True,
            "default_Lab_lookups_identical_at_printed_precision": True,
            "results": results, "commands": commands}, indent=2))


if __name__ == "__main__":
    unittest.main()
