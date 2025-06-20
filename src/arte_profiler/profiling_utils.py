import logging
import logging.handlers
import subprocess
from pathlib import Path
import pandas as pd
import platform
import importlib.resources
from typing import List, Optional
import numpy as np
from colour.utilities import tsplit
import hashlib

_loggers = {}

def _stream_handler() -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(
        logging.Formatter("%(asctime)s %(name)s - [%(levelname)s] %(message)s",
                          "%Y-%m-%d %H:%M:%S")
    )
    return h

def generate_logger(output_folder: Path, name: Optional[str] = None) -> tuple[logging.Logger, logging.Logger]:
    """
    Initializes and configures a logger that writes to both the console and a file.

    Parameters
    ----------
    output_folder : pathlib.Path
        Directory where the log file will be stored.
    name : str, optional
        Name of the logger.

    Returns
    -------
    tuple[logging.Logger, logging.Logger]
        (logger, command_logger)
    """
    # 1. choose a name ---------------------------------------------------
    if name is None:
        # Use folder name + short hash of absolute path for uniqueness and readability
        folder_name = output_folder.resolve().name
        folder_hash = hashlib.md5(str(output_folder.resolve()).encode()).hexdigest()[:6]
        name = f"profiling.{folder_name}.{folder_hash}"

    # 2. return cached version if it exists ------------------------------
    if name in _loggers:
        return _loggers[name], _loggers[f"{name}.command"]

    # 3. create fresh logger + handlers ----------------------------------
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "profiling.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=30 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s - [%(levelname)s] %(message)s",
                          "%Y-%m-%d %H:%M:%S")
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(_stream_handler())

    # command-only logger (keeps CLI print-out clean)
    command_logger = logging.getLogger(f"{name}.command")
    command_logger.setLevel(logging.DEBUG)
    command_logger.addHandler(file_handler)
    command_logger.propagate = False

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _loggers[name] = logger
    _loggers[f"{name}.command"] = command_logger
    logger.info("Logging initialised. Log file: %s", log_file)
    return logger, command_logger


def get_argyll_bin_path() -> str:
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
    # base_path = Path(__file__).parents[2] / "tools/argyllcms_v3.3.0"
    base_path = (
        importlib.resources.files("arte_profiler") / "tools" / "argyllcms_v3.3.0"
    )
    system = platform.system().lower()

    if "darwin" in system:  # macOS
        return str(base_path / "macos" / "bin")
    elif "linux" in system:  # Linux
        return str(base_path / "linux" / "bin")
    elif "windows" in system:  # Windows
        return str(base_path / "windows" / "bin")
    else:
        raise OSError(f"Unsupported platform: {system}")
    

def run_command(
    command: List[str],
    logger: logging.Logger,
    stdin_path: Optional[Path] = None,
    stdout_path: Optional[Path] = None,
) -> int:
    """
    Run a command, optionally redirecting stdin/stdout to files,
    and log stdout/stderr lines as they arrive.

    Parameters
    ----------
    command : list[str]
        Command and arguments to run.
    logger : logging.Logger
        Logger for output.
    stdin_path : pathlib.Path, optional
        File to use for stdin.
    stdout_path : pathlib.Path, optional
        File to use for stdout.

    Returns
    -------
    int
        Return code of the process.
    """
    stdin_f = open(stdin_path, "r") if stdin_path else None
    stdout_f = open(stdout_path, "w") if stdout_path else None

    proc = subprocess.Popen(
        command,
        stdin=stdin_f,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in proc.stdout:
        logger.info(line.rstrip())
        if stdout_f:
            stdout_f.write(line)

    for line in proc.stderr:
        logger.error(line.rstrip())

    if stdin_f:
        stdin_f.close()
    if stdout_f:
        stdout_f.close()

    return proc.wait()


def parse_file(file: Path) -> list:
    """
    Parse a .ti3 or .cie file and return the data.

    Parameters
    ----------
    file : pathlib.Path
        Path to the .ti3 or .cie file.

    Returns
    -------
    list
        Parsed data rows.
    """
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
    """
    Read a .ti3 file into a pandas DataFrame.

    Parameters
    ----------
    file : pathlib.Path
        Path to the .ti3 file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with patch data.
    """
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


def delta_L_CIE2000(L_1, L_2, textiles: bool = False) -> np.ndarray:
    """
    Returns the lightness component of the CIE 2000 color difference formula.

    Parameters
    ----------
    L_1 : array_like
        First set of CIELAB lightness values (L*)
    L_2 : array_like
        Second set of CIELAB lightness values (L*)
    textiles : bool, optional
        if set to True: k_L=2 instead of k_L=1.

    Returns
    -------
    numpy.ndarray
        Colour difference (ΔL*2000).
    
    Notes
    -----
    Based on `colour.difference.delta_e.delta_E_CIE2000` from the Colour library, 
    modified so that only the lightness component is returned.
    """

    k_L = 2 if textiles else 1

    l_bar_prime = 0.5 * (L_1 + L_2)

    delta_L_prime = L_2 - L_1

    s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
               np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))

    d_L = np.abs(delta_L_prime / (k_L * s_L))

    return d_L

def delta_Eab_CIE2000(ab_1, ab_2) -> np.ndarray:
    """
    Returns the CIE 2000 chroma-hue color difference (ΔE(a*b*)) between two sets 
    of a*, b* values.

    Parameters
    ----------
    ab_1 : array_like, shape (..., 2)
        First set of CIELAB chromaticity coordinates [a*, b*]
    ab_2 : array_like, shape (..., 2)
        Second set of CIELAB chromaticity coordinates [a*, b*]

    Returns
    -------
    numpy.ndarray
        Color difference ΔE(a*b*).
    
    Notes
    -----
    Based on `colour.difference.delta_e.delta_E_CIE2000` from the Colour library, 
    modified so that the lightness component is omitted.
    """

    k_C = 1
    k_H = 1

    a_1, b_1 = tsplit(ab_1)
    a_2, b_2 = tsplit(ab_2)

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)

    c_bar = 0.5 * (c_1 + c_2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a_1_prime = a_1 * (1 + g)
    a_2_prime = a_2 * (1 + g)
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)
    c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

    h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
    h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

    h_bar_prime = np.where(np.fabs(h_1_prime - h_2_prime) <= 180,
                           0.5 * (h_1_prime + h_2_prime),
                           (0.5 * (h_1_prime + h_2_prime + 360)))

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h_2_prime - h_1_prime
    delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_C_prime = c_2_prime - c_1_prime
    delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) *
                     np.sin(np.deg2rad(0.5 * delta_h_prime)))

    s_C = 1 + 0.045 * c_bar_prime
    s_H = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

    d_Eab =  np.sqrt(
        (delta_C_prime / (k_C * s_C)) ** 2 +
        (delta_H_prime / (k_H * s_H)) ** 2 +
        (delta_C_prime / (k_C * s_C)) * (delta_H_prime / (k_H * s_H)) * r_T)

    return d_Eab
