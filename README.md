# Arte-Profiler

## Overview

arte-profiler is a Python wrapper around ArgyllCMS that builds an ICC color profile from an image of a supported color card and produces a report evaluating color accuracy against Metamorfoze and FADGI imaging guidelines, making it useful for cultural heritage photography workflows.

## Features

- **Automatic Fiducial Detection**: Uses SIFT-based feature matching to locate fiducial marks.
- **Color Accuracy Evaluation**: Computes `ΔE_76` and `ΔE_2000` to assess color accuracy.
- **ICC Profile Generation**: Generates ICC profiles using **ArgyllCMS**.
- **Report Generation**: Creates a PDF report with visualizations and analysis.

## Installation

### **Prerequisites**
- Python **>=3.8**
- **ArgyllCMS** (included in the package)
- System dependencies for `pyvips`:
  - **macOS**: Install via Homebrew:  
    ```sh
    brew install vips
    ```
  - **Linux** (Debian/Ubuntu):  
    ```sh
    sudo apt install libvips42
    ```
  - **Windows**: Download and install from [libvips website](https://libvips.github.io/libvips/)

### **Install the Package**

To install the package, clone the repository and install it in an isolated environment:

`git clone https://github.com/amarrocc/arte-profiler.git`
`cd arte-profiler`
`pip install .`

## Usage

### Command-Line Interface (CLI)

You can run arte-profiler directly from the command line:

`arte-profiler --chart_tif path/to/image.tiff --chart_type charttype --out_icc path/to/output.icc -O path/to/output_folder`

For all available options, run:

`arte-profiler --help`

### Python API
You can also use arte-profiler as a Python library:

```sh
from profiling import ColorProfileBuilder

builder = ColorProfileBuilder(
    chart_tif="path/to/image.tiff",
    chart_type="ColorCheckerSG",
    out_icc="path/to/output.icc",
    folder="path/to/output_folder"
)

builder.run()
```

## Data Structure
The package includes:

- Reference Data (data/targets/): YAML file defining chart data.
- Profiles (data/profiles/): Predefined ICC profiles.
- Tools (tools/): Contains ArgyllCMS binaries and DejaVuSans font for report generation.

## License
To write.


