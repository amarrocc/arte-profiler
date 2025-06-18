#!/usr/bin/env bash

# This script demonstrates three different examples of using the 
# arte-profiler command line tool to perform profiling and evaluation.

set +x
echo "===================================================="
echo "EXAMPLE 1: Build a profile from a single chart image"
echo "===================================================="
echo "This example builds an ICC profile from a ColorCheckerSG chart"
echo "and then tests it on the same chart."
echo "This is primarily for confirmation of profile generation and application."
echo "It is recommended to use a different chart for proper evaluation."
echo "Input: sample_CCSG.tiff (ColorCheckerSG chart)"
echo "Output: Files saved to output/output_ex1/"
echo ""

(
  set -x
  arte-profiler \
    --build_tif sample_CCSG.tiff \
    --build_type ColorCheckerSG \
    --test_tif sample_CCSG.tiff \
    --test_type ColorCheckerSG \
    -O output/output_ex1
)

set +x
echo ""
echo "=============================================="
echo "EXAMPLE 2: Evaluating an existing ICC profile"
echo "=============================================="
echo "This example takes the ICC profile created in Example 1 and"
echo "evaluates its performance using a different chart (DT-NGT2)."
echo "Input: sample_DT-NGT2.tiff + previously created ICC profile"
echo "Output: Files saved to output/output_ex2/"
echo ""

(
  set -x
  arte-profiler \
    --test_tif sample_DT-NGT2.tiff \
    --test_type DT-NGT2 \
    --in_icc output/output_ex1/input_profile.icc \
    -O output/output_ex2
)

set +x
echo ""
echo "================================================================"
echo "EXAMPLE 3: Generate and evaluate a color profile in a single run"
echo "================================================================"
echo "This example demonstrates the recommended workflow:"
echo "Build a profile using one chart (ColorCheckerSG) and then"
echo "evaluate its accuracy using a different chart (DT-NGT2)."
echo "Note: a single image containg two charts can also be used instead of two separate images."
echo "Input: sample_CCSG.tiff (for building) + sample_DT-NGT2.tiff (for testing)"
echo "Output: Files saved to output/output_ex3/"
echo ""

(
  set -x
  arte-profiler \
    --build_tif sample_CCSG.tiff \
    --build_type ColorCheckerSG \
    --test_tif sample_DT-NGT2.tiff \
    --test_type DT-NGT2 \
    -O output/output_ex3
)

set +x
echo ""
echo "=============================================="
echo "All examples completed!"
echo "=============================================="
echo "Check the output folders for generated reports and analysis files:"
echo "- output/output_ex1/: Build a profile from a single chart image"
echo "- output/output_ex2/: Evaluating an existing ICC profile"
echo "- output/output_ex3/: Generate and evaluate a color profile in a single run"
echo ""


