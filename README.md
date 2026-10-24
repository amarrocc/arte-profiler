# Arte-Profiler

**Version: 0.1-beta**  

## Overview

**arte-profiler** is a Python-based tool that simplifies the usage and extends the functionality of Argyll CMS for camera profiling and color accuracy evaluation. The software is designed primarily for use in cultural heritage digitization but can be useful also for other applications requiring precise color reproduction. The tool provides both a command-line interface (CLI) and a programmatic API, enabling integration into automated imaging workflows or standalone usage.

The program supports three primary use cases:

1. **Generating a color profile** from an image of a supported color chart.
2. **Evaluating an existing ICC profile** using an image of a supported color chart.
3. **Generating and evaluating a color profile** in a single run, either from an image that contains two different supported charts or from two separate images. In this last case, one chart is used to generate the color profile, while the other one assesses its accuracy.

In all cases, arte-profiler generates a structured **PDF report** summarizing the results based on the **Metamorfoze** and **FADGI** imaging guidelines.

## Supported targets

arte-profiler supports the following standard color charts:

- **ColorChecker Digital SG**
- **ColorChecker Passport**
- **Digital Transitions Next Generation Target, Version 2 (DT-NGT2)**

The tool bundles all necessary reference data for target recognition and profiling, including:

- A reference image of the target
- The pixel coordinates of fiducial marks on the reference image
- Data defining the geometric layout of the target
- Generic L\*a\*b\* reference values for each patch (sourced from manufacturer data)

While these generic L\*a\*b\* values provide a practical baseline for profiling, **using chart-specific measurements is recommended for optimal accuracy.** New targets can also be included by adding their reference data.

## Installation

### Prerequisites

- Python 3.11
- Argyll CMS is bundled in this repository (version 3.3.0).

### Install arte-profiler

1. Clone the GitHub repository:

    ```bash
    git clone https://github.com/amarrocc/arte-profiler
    cd arte-profiler
    ```

2. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv arte-profiler
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```

3. Install the package in editable mode:

    ```bash
    pip install -e .
    ```

    This will install all dependencies specified in `pyproject.toml`.

## GUI development

The optional **Arte-Profiler** NiceGUI interface guides users through two
workflows in the local browser:

- **Create and evaluate a profile:** select a profiling TIFF and target type,
  then choose evaluation reference data and an output directory in the
  Evaluation step. An ICC profile and creation report are always included.
- **Evaluate an existing profile:** select an ICC profile, evaluation TIFF,
  target, and output directory.

Supported targets are listed on the welcome screen from the repository manifest.
Custom CIE files containing chart-specific measurements are optional and appear
under **Advanced options**; leave these blank to use chart generic reference
values. After the profiling inputs, choose what reference data is available:
only the profiling target, another target in the same TIFF, or a target in a
separate TIFF. Only relevant fields are shown. The same-TIFF choice reuses the
profiling image while retaining a separate evaluation target and reference.
Back navigation preserves entered values, including a separate TIFF path when
temporarily switching to the same-TIFF choice.
When only the profiling target is available, the Evaluation step displays a
short recommendation to use a second different target.

Both workflows end with a review of the selected inputs and expected outputs.
The review screen also contains Processing log and Results areas. The Run
button remains disabled: scientific processing is not connected yet, and the
interface does not create profiles, reports, or output directories.

Use Python 3.11. From the repository root, create and activate a development
environment (or activate an existing Python 3.11 environment):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell or `.venv\Scripts\activate.bat`
in Command Prompt.

Install the package with its optional GUI dependency:

```bash
python -m pip install -e ".[gui]"
```

Launch the interface:

```bash
python -m arte_profiler.gui
```

The equivalent console command is `arte-profiler-gui`. The application opens
in your browser at <http://127.0.0.1:8080> and listens only on `127.0.0.1`.
Stop the server with Ctrl-C in the launching terminal. Automatic reload is
disabled; restart the server after editing GUI code.

Enter local filesystem paths or use **Browse** to choose TIFF, CIE, and ICC
files or an existing output directory. The picker browses the computer running
Arte-Profiler without uploading or copying files. Click a folder to open it or
a matching file to select it; for output, use **Select this folder**. The
Location field also accepts a folder path, including Windows drive paths.
New output directory paths can be entered directly without creating them.

Supported targets are loaded from the existing manifest. Leaving a CIE
reference blank selects bundled reference values when processing is added.
The creation report verifies profile generation and application on the
profiling target. An additional target with different color patches is
recommended for assessing accuracy across colors. A separate TIFF containing
the same patch layout is not automatically an independent evaluation across
colors; the interface calls this out when the target types match.

The expected-output summary always includes the ICC profile and creation
report for creation. Either additional-target choice adds an evaluation
report; choosing only the profiling target performs no additional evaluation.
Evaluating an existing ICC profile produces an evaluation report only. These
are planned outputs until processing is connected.

The existing CLI remains available as `arte-profiler`, and `pip install -e .`
continues to install the package without NiceGUI. Importing the GUI still loads
the package's existing scientific dependencies through `arte_profiler.__init__`.

To run the GUI smoke tests, install the test tools in the same environment:

```bash
python -m pip install pytest pytest-asyncio
python -m pytest tests/test_gui_smoke.py -q
```

These tests use NiceGUI's simulated user, so no browser driver is required.
They check both guided workflows, conditional fields, review summaries,
retained paths, advanced references, scientific explanations, file/folder
browsing, and the disabled Run button.
The GUI tests are skipped when NiceGUI or pytest-asyncio is not installed.

## Command-Line Usage

After installation, a script named `arte-profiler` will be available in your environment. You can run:

```bash
arte-profiler --help
```

### Basic Examples

#### 1. **Generate a profile from a single chart image**

```bash
arte-profiler \
  --build_tif path/to/chart_image.tiff \
  --build_type chart_type \
  --build_cie path/to/custom_chart.cie \  # <-- optionally specify custom reference Lab values
  --test_tif path/to/chart_image.tiff \
  --test_type chart_type \
  --test_cie path/to/custom_chart.cie \  # <-- optionally specify custom reference Lab values
  -O output_folder
```

This command:
1. Builds a new ICC profile from `chart_image.tiff` which contains a chart of type `chart_type`.
2. Evaluates it against the same image (to verify that the profile was correctly generated and applied).
3. Saves a report and intermediate files in `output_folder`.

**N.B.:** The generated report primarily serves to verify that the profile was correctly generated and applied, rather than to provide a rigorous evaluation of its performance. **Assessing profile accuracy using an independent target** (i.e. one with different patches than the one used for profiling) **is strongly recommended** to mitigate the risk of overestimating color accuracy.

#### 2. **Evaluate an existing ICC profile**

```bash
arte-profiler \
  --test_tif path/to/chart_image.tiff \
  --test_type chart_type \
  --test_cie path/to/custom_chart.cie \  # <-- optionally specify custom reference Lab values
  --in_icc path/to/existing_profile.icc \
  -O output_folder
```
This command:
1. Evaluates `existing_profile.icc` using `chart_image.tiff` which contains a chart of type `chart_type`.
2. Saves a report and intermediate files in `output_folder`.

#### 3. **Generate and evaluate a color profile in a single run**

```bash
arte-profiler \
  --build_tif path/to/chartA_image.tiff \
  --build_type chartA_type \
  --build_cie path/to/custom_chartA.cie \  # <-- optionally specify custom reference Lab values
  --test_tif path/to/chartB_image.tiff \
  --test_type chartB_type \
  --test_cie path/to/custom_chartB.cie \  # <-- optionally specify custom reference Lab values

  -O output_folder
```
This command:
1. Builds a new ICC profile from `chartA_image.tif` which contains a chart of type `chartA_type`.
2. Evaluates the newly build input profile against the same image (to verify that the profile was correctly generated and applied).
3. Saves a (first) creation report and intermediate files in `output_folder`.
4. Evaluates the newly build input profile against `chartB_image.tif` which contains a different chart (`chartB_type`).
5. Saves a (second) evaluation report and and intermediate files in `output_folder`.

**Note**: In this example, the `build_tif` and `test_tif` parameters refer to two separate image files; alternatively, a single image (i.e. `build_tif` and `test_tif` are the same file) that contains two different supported charts can be used.

### Example script

For more complete CLI usage scenarios, see the example script [`examples/cli_example.sh`](examples/cli_example.sh)

## Programmatic Usage

### Building an ICC Profile

```python
from arte_profiler.profiling import ProfileCreator

creator = ProfileCreator(
    chart_tif="path/to/build_image.tiff ",
    chart_type="build_chart_type",
    folder="output_folder"
)
creator.build_profile()
```

### Evaluating a Profile

```python
from arte_profiler.profiling import ProfileEvaluator

evaluator = ProfileEvaluator(
    chart_tif="test_image.tiff",
    chart_type="test_chart_type",
    in_icc="my_profile.icc",
    folder="output_folder"
)
evaluator.evaluate_profile()
```

### Example script

For more complete API usage scenarios, see the example script [`examples/sample_api_usage.py`](examples/sample_api_usage.py)


## Output Reports

arte-profiler generates structured PDF reports that include:
- ΔE*₀₀ for each patch in a chart
- A histogram of the ΔE*₀₀ values
- Compliance checks for Metamorfoze and FADGI guidelines (color accuracy, oecf, white balance).
- An appendix with information relative to the patches values extraction

This is an example extract from a generated PDF report:

![Example report output](assets/report_example.png)

## Dependencies

arte-profiler requires the following dependencies:
- `colour-science`
- `matplotlib`
- `numpy`
- `opencv-python`
- `pandas`
- `imageio`
- `pyyaml`
- `reportlab`
- `seaborn`
- `shapely`

## License

arte-profiler is released under **GNU General Public License**. See `LICENSE` for details.

## Authors

- **Alessandra Marrocchesi** - *Lead Developer*
- **Robert G. Erdmann** - *Contributor*

## Acknowledgments

arte-profiler integrates ArgyllCMS, which is distributed with the package.

## Contributions

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/amarrocc/arte-profiler).





