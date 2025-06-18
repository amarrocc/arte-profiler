from arte_profiler.profiling import ProfileCreator, ProfileEvaluator
from pathlib import Path

def main() -> None:
    """
    Example usage scenarios for arte-profiler.

    Demonstrates:
    1. Building and evaluating a profile on the same chart.
    2. Evaluating an existing ICC profile on a chart.
    3. Building a profile from one chart and evaluating it on a different chart.
    """
    EXAMPLES_DIR = Path(__file__).parent

    # Scenario 1: Build and evaluate a profile on the same chart (not recommended for real evaluation)
    print("=== Scenario 1: Build and evaluate on the same chart (sample_CCSG.tiff, ColorCheckerSG) ===")
    OUTPUT_DIR1 = EXAMPLES_DIR / "output" / "scenario1"
    OUTPUT_DIR1.mkdir(parents=True, exist_ok=True)
    chart_tif = EXAMPLES_DIR / "sample_CCSG.tiff"
    chart_type = "ColorCheckerSG"
    builder = ProfileCreator(
        chart_tif=chart_tif,
        chart_type=chart_type,
        folder=OUTPUT_DIR1,
    )
    profile_path = builder.build_profile()
    evaluator = ProfileEvaluator(
        chart_tif=chart_tif,
        chart_type=chart_type,
        in_icc=profile_path,
        folder=OUTPUT_DIR1,
        patch_data=builder.df,
    )
    evaluator.evaluate_profile(report_title="Profile Creation Report", report_filename="profile_creation_report_CCSG.pdf")

    # Scenario 2: Evaluate an existing ICC profile on a chart (sample_DT-NGT2.tiff, DT-NGT2)
    print("=== Scenario 2: Evaluate an existing ICC profile (sample_DT-NGT2.tiff, DT-NGT2) ===")
    OUTPUT_DIR2 = EXAMPLES_DIR / "output" / "scenario2"
    OUTPUT_DIR2.mkdir(parents=True, exist_ok=True)
    test_tif = EXAMPLES_DIR / "sample_DT-NGT2.tiff"
    test_type = "DT-NGT2"
    existing_icc = profile_path  # For demonstration, we reuse the profile just built
    evaluator2 = ProfileEvaluator(
        chart_tif=test_tif,
        chart_type=test_type,
        in_icc=existing_icc,
        folder=OUTPUT_DIR2,
    )
    evaluator2.evaluate_profile(report_title="Profile Evaluation Report", report_filename="profile_evaluation_report_DT-NGT2.pdf")

    # Scenario 3: Build a profile from one chart and evaluate it on a different chart
    print("=== Scenario 3: Build and evaluate using two different charts in one image (sample_dual_capture.tiff) ===")
    OUTPUT_DIR3 = EXAMPLES_DIR / "output" / "scenario3"
    OUTPUT_DIR3.mkdir(parents=True, exist_ok=True)
    dual_tif = EXAMPLES_DIR / "sample_dual_capture.tiff"
    build_type = "ColorCheckerSG"
    test_type = "DT-NGT2"
    # Build profile using ColorCheckerSG region
    builder3 = ProfileCreator(
        chart_tif=dual_tif,
        chart_type=build_type,
        folder=OUTPUT_DIR3,
    )
    profile_path3 = builder3.build_profile()
    # Evaluate profile using ColorCheckerSG region (for info)
    evaluator3_creation = ProfileEvaluator(
        chart_tif=dual_tif,
        chart_type=build_type,
        in_icc=profile_path3,
        folder=OUTPUT_DIR3,
        patch_data=builder3.df,
    )
    evaluator3_creation.evaluate_profile(report_title="Profile Creation Report (Dual Capture)", report_filename="profile_creation_report_dual.pdf")
    # Evaluate profile using DT-NGT2 region
    evaluator3 = ProfileEvaluator(
        chart_tif=dual_tif,
        chart_type=test_type,
        in_icc=profile_path3,
        folder=OUTPUT_DIR3,
    )
    evaluator3.evaluate_profile(report_title="Profile Evaluation Report (Dual Capture)", report_filename="profile_evaluation_report_dual.pdf")


if __name__ == "__main__":
    main()