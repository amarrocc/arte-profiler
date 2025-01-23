import logging
import subprocess
from pathlib import Path
import pandas as pd
import platform

def get_argyll_bin_path():
    """
    Determine the path to the ArgyllCMS `bin` directory based on the current platform.

    Returns
    -------
    str
        Path to the appropriate ArgyllCMS `bin` directory.

    Raises
    ------
    OSError
        If the platform is not supported.
    """
    base_path = Path(__file__).parent.parent / "tools/argyllcms_v3.3.0"
    system = platform.system().lower()

    if "darwin" in system:  # macOS
        return str(base_path / "macos" / "bin")
    elif "linux" in system:  # Linux
        return str(base_path / "linux" / "bin")
    elif "windows" in system:  # Windows
        return str(base_path / "windows" / "bin")
    else:
        raise OSError(f"Unsupported platform: {system}")
    
def run_command(command: list, logger: logging.Logger):
    """Run a command and log the stdout and stderr as it becomes available."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    for line in iter(process.stdout.readline, ""):
        logger.info(line.strip())

    for line in iter(process.stderr.readline, ""):
        logger.error(line.strip())

    return process.wait()


def parse_file(file: Path):
    """Parse a .ti3 or .cie file and return the data."""
    data = []
    with open(file) as f:
        for line in f:
            if line.strip() == "BEGIN_DATA":
                break
        for line in f:
            if line.strip() == "END_DATA":
                break
            data.append(line.split())
    return data


def ti3_to_dataframe(file: Path) -> pd.DataFrame:
    """Read a .ti3 file into a DataFrame."""
    df = pd.DataFrame(
        parse_file(file),
        columns=[
            "SAMPLE_ID",
            "XYZ_X",
            "XYZ_Y",
            "XYZ_Z",
            "RGB_R",
            "RGB_G",
            "RGB_B",
            "STDEV_R",
            "STDEV_G",
            "STDEV_B",
        ],
    )

    df["row"] = df.SAMPLE_ID.str[1:].astype(int)
    df["col"] = df.SAMPLE_ID.str[0]
    df = df[[df.columns[0], *df.columns[-2:], *df.columns[1:-2]]]
    df.set_index("SAMPLE_ID", inplace=True)

    for col in df.columns[2:]:
        df[col] = df[col].astype(float)

    return df
