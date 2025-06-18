#!/usr/bin/env bash

# Enable verbose mode to show commands as they are executed
set -x

# This script demonstrates three different examples of using the 
# arte-profiler command line tool to perform profiling and analysis

# Example 1: Build a profile from a single chart image:
#    Note: This builds and tests the profile on the same chart. The 
#    report mainly confirms correct generation and application, not 
#    accuracy. For proper evaluation, use a separate chart (see case 3) 
#    to avoid overestimating performance.

echo "=============================================="
echo "EXAMPLE 1: Building and testing a color profile"
echo "=============================================="
echo "This example builds an ICC profile from a ColorCheckerSG chart"
echo "and then tests it on the same chart for basic validation."
echo "Input: sample_CCSG.tiff (ColorCheckerSG chart)"
echo "Output: Files saved to output_folder1/"
echo ""

arte-profiler \
  --build_tif sample_CCSG.tiff \
  --build_type ColorCheckerSG \
  --test_tif sample_CCSG.tiff \
  --test_type ColorCheckerSG \
  -O output_folder1


echo ""
echo "=============================================="
echo "EXAMPLE 2: Evaluating an existing ICC profile"
echo "=============================================="
echo "This example takes the ICC profile created in Example 1 and"
echo "evaluates its performance using a different chart (DT-NGT2)."
echo "Input: sample_DT-NGT2.tiff + previously created ICC profile"
echo "Output: Files saved to output_folder2/"
echo ""

# Example 2: Evaluate an existing ICC profile, in this case the one
#    generated from the first example, using a different chart image.
   arte-profiler \
     --test_tif sample_DT-NGT2.tiff \
     --test_type DT-NGT2 \
     --in_icc output_folder1/input_profile.icc \
     -O output_folder2


echo ""
echo "=============================================="
echo "EXAMPLE 3: Complete workflow with dual charts"
echo "=============================================="
echo "This example demonstrates the recommended workflow:"
echo "Build a profile using one chart (ColorCheckerSG) and then"
echo "evaluate its accuracy using a different chart (DT-NGT2)."
echo "Input: sample_CCSG.tiff (for building) + sample_DT-NGT2.tiff (for testing)"
echo "Output: Files saved to output_folder3/"
echo ""

# Example 3: Generate and evaluate a color profile in a single run:
  arte-profiler \
    --build_tif sample_CCSG.tiff \
    --build_type ColorCheckerSG \
    --test_tif sample_DT-NGT2.tiff \
    --test_type DT-NGT2 \
    -O output_folder3


echo ""
echo "=============================================="
echo "All examples completed!"
echo "=============================================="
echo "Check the output folders for generated reports and analysis files:"
echo "- output_folder1/: Profile creation and self-validation"
echo "- output_folder2/: Profile evaluation on different chart"
echo "- output_folder3/: Complete dual-chart workflow"
echo ""


