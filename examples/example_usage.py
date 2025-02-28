from arte_profiler import ProfileCreator
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent

# Define paths (modify if needed)
chart_tif =  EXAMPLES_DIR / "sample_image.tiff" # Sample image of a color chart
out_icc = "eciRGB_v2"  #output icc profile
output_folder = EXAMPLES_DIR / "output"  # Where to save logs/reports

output_folder.mkdir(parents=True, exist_ok=True)

# Initialize and run
builder = ProfileCreator(
    chart_tif=chart_tif,
    chart_type="ColorCheckerSG",
    folder=output_folder,
)

builder.build_profile()