"""Command and workflow regressions; no scientific subprocesses run here."""

import contextlib
import io
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from arte_profiler import profiling


class ProfileOptionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = self.root / "chart.tiff"
        self.image.touch()  # Constructor checks existence; these tests never read pixels.
        logger_patch = patch.object(profiling.profiling_utils, "generate_logger", return_value=(Mock(), Mock()))
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        process_patch = patch.object(profiling.profiling_utils.subprocess, "Popen")
        self.popen = process_patch.start()
        self.addCleanup(process_patch.stop)
        self.popen.return_value.stdout = []
        self.popen.return_value.stderr = []
        self.popen.return_value.wait.return_value = 0

    def creator(self, **options):
        creator = profiling.ProfileCreator(self.image, "ColorCheckerSG", folder=self.root, **options)
        creator.chart_ti3 = self.root / "measurements.ti3"
        return creator

    def legacy_command(self, creator, name="input_profile.icc"):
        # Frozen original argv, deliberately independent of the option constants.
        return [str(Path(creator.argyll_bin_path) / "colprof"), "-v", "-Z", "a", "-ua",
                "-a", "g", "-b", "n", "-q", "m", "-O",
                str(self.root / name), str(creator.chart_ti3.with_suffix(""))]

    def test_omitted_options_produce_exact_original_command(self):
        creator = self.creator()
        self.assertEqual(creator.icc_from_ti3(), self.root / "input_profile.icc")
        self.popen.assert_called_once_with(self.legacy_command(creator), stdin=None,
                                           stdout=profiling.profiling_utils.subprocess.PIPE,
                                           stderr=profiling.profiling_utils.subprocess.PIPE, text=True)

    def test_original_positional_constructor_and_method_calls(self):
        creator = profiling.ProfileCreator(self.image, "ColorCheckerSG", None, self.root, "legacy")
        creator.chart_ti3 = self.root / "measurements.ti3"
        with patch.object(creator, "detect_and_extract") as extract:
            fiducials = list(range(8))
            self.assertEqual(creator.build_profile(fiducials, "legacy.icc"), self.root / "legacy.icc")
            extract.assert_called_once_with(fiducial=fiducials)
        self.assertEqual(self.popen.call_args.args[0], self.legacy_command(creator, "legacy.icc"))
        self.popen.reset_mock()
        creator.icc_from_ti3("another.icc")
        self.assertEqual(self.popen.call_args.args[0], self.legacy_command(creator, "another.icc"))

    def test_all_fifteen_combinations_change_only_algorithm_and_complexity(self):
        for algorithm, complexity in itertools.product("gsmlx", "lmh"):
            with self.subTest(algorithm=algorithm, complexity=complexity):
                self.popen.reset_mock()
                creator = self.creator(profile_algorithm=algorithm, profile_complexity=complexity)
                expected = self.legacy_command(creator, "custom.icc")
                expected[6], expected[10] = algorithm, complexity
                self.assertEqual(creator.icc_from_ti3("custom.icc"), self.root / "custom.icc")
                self.assertEqual(creator.in_icc, self.root / "custom.icc")
                self.popen.assert_called_once()
                self.assertEqual(self.popen.call_args.args[0], expected)

    def test_each_option_can_be_supplied_independently(self):
        for options, algorithm, complexity in (({"profile_algorithm": "s"}, "s", "m"),
                                             ({"profile_complexity": "h"}, "g", "h")):
            with self.subTest(options=options):
                creator = self.creator(**options)
                creator.icc_from_ti3()
                expected = self.legacy_command(creator)
                expected[6], expected[10] = algorithm, complexity
                self.assertEqual(self.popen.call_args.args[0], expected)

    def test_invalid_constructor_options_fail_before_file_or_process_access(self):
        for name, invalid_values in (("profile_algorithm", ("", "G", "X", "z", "g -ua", None, 1, [])),
                                     ("profile_complexity", ("", "u", "H", "slow", None, 1, []))):
            for invalid in invalid_values:
                with self.subTest(name=name, value=invalid):
                    with patch.object(profiling.BaseColorManager, "__init__") as base:
                        with self.assertRaisesRegex(ValueError, f"Invalid {name}.*Choose one of"):
                            profiling.ProfileCreator("missing.tiff", "ColorCheckerSG", **{name: invalid})
                        base.assert_not_called()
        self.popen.assert_not_called()

    def test_modified_options_are_revalidated_before_processing(self):
        for method in ("build_profile", "icc_from_ti3"):
            creator = self.creator()
            creator.profile_complexity = "u"
            with patch.object(creator, "detect_and_extract") as extract:
                with self.assertRaisesRegex(ValueError, "profile_complexity"):
                    getattr(creator, method)()
                extract.assert_not_called()
        self.popen.assert_not_called()

    def test_missing_measurements_and_subprocess_failure_keep_existing_errors(self):
        creator = self.creator()
        del creator.chart_ti3
        with self.assertRaisesRegex(RuntimeError, "extraction"):
            creator.icc_from_ti3()
        self.popen.assert_not_called()
        creator.chart_ti3 = self.root / "measurements.ti3"
        self.popen.return_value.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "colprof command failed"):
            creator.icc_from_ti3()
        self.assertFalse(hasattr(creator, "in_icc"))

    def test_build_uses_configured_options_and_preserves_extraction_arguments(self):
        creator = self.creator(profile_algorithm="x", profile_complexity="l")
        with patch.object(creator, "detect_and_extract") as extract:
            fiducials = list(range(8))
            creator.build_profile(fiducials, "lut.icc")
            extract.assert_called_once_with(fiducial=fiducials)
        expected = self.legacy_command(creator, "lut.icc")
        expected[6], expected[10] = "x", "l"
        self.assertEqual(self.popen.call_args.args[0], expected)


class ProfileOptionsCLITests(unittest.TestCase):
    BUILD = ["--build_tif", "build.tiff", "--build_type", "ColorCheckerSG",
             "--test_tif", "test.tiff", "--test_type", "DT-NGT2", "-O", "results"]
    EVALUATE = ["--in_icc", "existing.icc", "--test_tif", "test.tiff",
                "--test_type", "DT-NGT2", "-O", "results"]

    def parse(self, argv):
        with patch("sys.argv", ["arte-profiler", *argv]):
            return profiling.parse_args()

    def test_cli_defaults_and_all_supported_choices(self):
        args = self.parse(self.BUILD)
        self.assertEqual((args.profile_algorithm, args.profile_complexity), ("g", "m"))
        for algorithm, complexity in itertools.product("gsmlx", "lmh"):
            with self.subTest(algorithm=algorithm, complexity=complexity):
                args = self.parse(self.BUILD + ["--profile-algorithm", algorithm, "--profile-complexity", complexity])
                self.assertEqual((args.profile_algorithm, args.profile_complexity), (algorithm, complexity))

    def test_cli_rejects_invalid_values_with_choices(self):
        for flag, invalid in (("--profile-algorithm", "G"), ("--profile-algorithm", "z"),
                              ("--profile-complexity", "u"), ("--profile-complexity", "H")):
            with self.subTest(flag=flag, invalid=invalid), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as exit_status:
                    self.parse(self.BUILD + [flag, invalid])
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn("invalid choice", error.getvalue())
                self.assertIn(flag, error.getvalue())

    def test_main_forwards_all_choices_and_preserves_both_reports(self):
        for algorithm, complexity in itertools.product("gsmlx", "lmh"):
            with self.subTest(algorithm=algorithm, complexity=complexity):
                with patch("sys.argv", ["arte-profiler", *self.BUILD, "--profile-algorithm", algorithm,
                                        "--profile-complexity", complexity]), \
                        patch.object(profiling, "ProfileCreator") as creator, \
                        patch.object(profiling, "ProfileEvaluator") as evaluator:
                    profiling.main()
                    creator.assert_called_once_with(chart_tif="build.tiff", chart_type="ColorCheckerSG",
                        chart_cie=None, folder="results", logger_name="profiling",
                        profile_algorithm=algorithm, profile_complexity=complexity)
                    creator.return_value.build_profile.assert_called_once_with(None)
                    self.assertEqual(evaluator.call_count, 2)
                    first, second = evaluator.call_args_list
                    self.assertIs(first.kwargs["patch_data"], creator.return_value.df)
                    self.assertIsNone(second.kwargs["patch_data"])
                    self.assertEqual(second.kwargs["chart_type"], "DT-NGT2")
                    self.assertEqual(evaluator.return_value.evaluate_profile.call_args_list[0].kwargs,
                        {"report_title": "Profile Creation Report", "report_filename": "profile_creation_report.pdf"})
                    self.assertEqual(evaluator.return_value.evaluate_profile.call_args_list[1].kwargs,
                        {"report_title": "Profile Evaluation Report", "report_filename": "profile_evaluation_report.pdf"})

    def test_original_cli_build_command_and_same_target_creation_report(self):
        argv = ["--build_tif", "build.tiff", "--build_type", "ColorCheckerSG",
                "--test_tif", "build.tiff", "--test_type", "ColorCheckerSG"]
        with patch("sys.argv", ["arte-profiler", *argv]), \
                patch.object(profiling, "ProfileCreator") as creator, \
                patch.object(profiling, "ProfileEvaluator") as evaluator:
            profiling.main()
            self.assertEqual(creator.call_args.kwargs["profile_algorithm"], "g")
            self.assertEqual(creator.call_args.kwargs["profile_complexity"], "m")
            evaluator.assert_called_once()
            evaluator.return_value.evaluate_profile.assert_called_once_with(None,
                report_title="Profile Creation Report", report_filename="profile_creation_report.pdf")

    def test_evaluation_only_never_uses_build_options(self):
        for extra in ([], ["--profile-algorithm", "x", "--profile-complexity", "h"]):
            with self.subTest(extra=extra), patch("sys.argv", ["arte-profiler", *self.EVALUATE, *extra]), \
                    patch.object(profiling, "ProfileCreator") as creator, \
                    patch.object(profiling, "ProfileEvaluator") as evaluator:
                profiling.main()
                creator.assert_not_called()
                evaluator.assert_called_once_with(chart_tif="test.tiff", chart_type="DT-NGT2", chart_cie=None,
                    in_icc="existing.icc", folder="results", patch_data=None, logger_name="profiling")
                evaluator.return_value.evaluate_profile.assert_called_once_with(None,
                    report_title="Profile Evaluation Report", report_filename="profile_evaluation_report.pdf")


if __name__ == "__main__":
    unittest.main()
