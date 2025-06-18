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

arte-profiler \
  --build_tif sample_CCSG.tiff \
  --build_type ColorCheckerSG \
  --test_tif sample_CCSG.tiff \
  --test_type ColorCheckerSG \
  -O output_folder1


# Example 2: Evaluate an existing ICC profile, in this case the one
#    generated from the first example, using a different chart image.
   arte-profiler \
     --test_tif sample_DT-NGT2.tiff \
     --test_type DT-NGT2 \
     --in_icc output_folder1/input_profile.icc \
     -O output_folder2


# Example 3: Generate and evaluate a color profile in a single run:
  arte-profiler \
    --build_tif sample_CCSG.tiff \
    --build_type ColorCheckerSG \
    --test_tif sample_DT-NGT2.tiff \
    --test_type DT-NGT2 \
    -O output_folder3


