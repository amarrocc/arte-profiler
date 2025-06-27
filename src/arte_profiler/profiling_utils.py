import logging
import logging.handlers
import subprocess
from pathlib import Path
import pandas as pd
import platform
import importlib.resources
from typing import List, Optional
import numpy as np
from colour.utilities import tsplit, to_domain_100, as_float
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

def delta_E_CIE2000(
    Lab_1, Lab_2, textiles: bool = False, SL1: bool = False
):
    """
    Returns the CIE 2000 colour difference.

    Parameters
    ----------
    Lab_1 : array_like, shape (..., 3)
        First set of LAB values
    Lab_2 : array_like, shape (..., 3)
        Second set of LAB values
    textiles
        Textiles application specific parametric factors.
        :math:`k_L=2,\\ k_C=k_H=1` weights are used instead of
        :math:`k_L=k_C=k_H=1`.
    SL1
        If set to True, SL=1 is used instead of the default SL value.

    Returns
    -------
    numpy.ndarray
        Color difference ΔE*.

    Notes
    -----
    This is `colour.difference.delta_e.delta_E_CIE2000` from the Colour library, 
    with the added option to set `SL1` to True, which uses SL=1 instead of 
    the default SL value.
    """

    L_1, a_1, b_1 = tsplit(to_domain_100(Lab_1))
    L_2, a_2, b_2 = tsplit(to_domain_100(Lab_2))

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    C_1_ab = np.hypot(a_1, b_1)
    C_2_ab = np.hypot(a_2, b_2)

    C_bar_ab = (C_1_ab + C_2_ab) / 2
    C_bar_ab_7 = C_bar_ab**7

    G = 0.5 * (1 - np.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25**7)))

    a_p_1 = (1 + G) * a_1
    a_p_2 = (1 + G) * a_2

    C_p_1 = np.hypot(a_p_1, b_1)
    C_p_2 = np.hypot(a_p_2, b_2)

    h_p_1 = np.where(
        np.logical_and(b_1 == 0, a_p_1 == 0),
        0,
        np.degrees(np.arctan2(b_1, a_p_1)) % 360,
    )
    h_p_2 = np.where(
        np.logical_and(b_2 == 0, a_p_2 == 0),
        0,
        np.degrees(np.arctan2(b_2, a_p_2)) % 360,
    )

    delta_L_p = L_2 - L_1

    delta_C_p = C_p_2 - C_p_1

    h_p_2_s_1 = h_p_2 - h_p_1
    C_p_1_m_2 = C_p_1 * C_p_2
    delta_h_p = np.select(
        [
            C_p_1_m_2 == 0,
            np.fabs(h_p_2_s_1) <= 180,
            h_p_2_s_1 > 180,
            h_p_2_s_1 < -180,
        ],
        [
            0,
            h_p_2_s_1,
            h_p_2_s_1 - 360,
            h_p_2_s_1 + 360,
        ],
    )

    delta_H_p = 2 * np.sqrt(C_p_1_m_2) * np.sin(np.deg2rad(delta_h_p / 2))

    L_bar_p = (L_1 + L_2) / 2

    C_bar_p = (C_p_1 + C_p_2) / 2

    a_h_p_1_s_2 = np.fabs(h_p_1 - h_p_2)
    h_p_1_a_2 = h_p_1 + h_p_2
    h_bar_p = np.select(
        [
            C_p_1_m_2 == 0,
            a_h_p_1_s_2 <= 180,
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 < 360),
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
        1
        - 0.17 * np.cos(np.deg2rad(h_bar_p - 30))
        + 0.24 * np.cos(np.deg2rad(2 * h_bar_p))
        + 0.32 * np.cos(np.deg2rad(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.deg2rad(4 * h_bar_p - 63))
    )

    delta_theta = 30 * np.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p**7
    R_C = 2 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + 25**7))

    L_bar_p_2 = (L_bar_p - 50) ** 2

    if SL1:
        S_L = 1
    else:
        S_L = 1 + ((0.015 * L_bar_p_2) / np.sqrt(20 + L_bar_p_2))

    S_C = 1 + 0.045 * C_bar_p

    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -np.sin(np.deg2rad(2 * delta_theta)) * R_C

    d_E = np.sqrt(
        (delta_L_p / (k_L * S_L)) ** 2
        + (delta_C_p / (k_C * S_C)) ** 2
        + (delta_H_p / (k_H * S_H)) ** 2
        + R_T * (delta_C_p / (k_C * S_C)) * (delta_H_p / (k_H * S_H))
    )

    return as_float(d_E)

def delta_L_CIE2000(L_1, L_2, textiles: bool = False, SL1: bool=False) -> np.ndarray:
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
    SL1 : bool, optional
        if set to True, SL=1 is used instead of the default SL value.

    Returns
    -------
    numpy.ndarray
        Colour difference (ΔL*2000).
    
    Notes
    -----
    Based on `colour.difference.delta_e.delta_E_CIE2000` from the Colour library, 
    modified so that only the lightness component is returned.
    """
    L_1 = to_domain_100(L_1)
    L_2 = to_domain_100(L_2)

    k_L = 2 if textiles else 1

    delta_L_p = L_2 - L_1
    L_bar_p = (L_1 + L_2) / 2
    L_bar_p_2 = (L_bar_p - 50) ** 2
    if SL1:
        S_L = 1
    else:
        S_L = 1 + ((0.015 * L_bar_p_2) / np.sqrt(20 + L_bar_p_2))

    d_L = np.abs(delta_L_p / (k_L * S_L))

    return as_float(d_L)

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
    a_1, b_1 = tsplit(to_domain_100(ab_1))
    a_2, b_2 = tsplit(to_domain_100(ab_2))

    k_C = 1
    k_H = 1

    C_1_ab = np.hypot(a_1, b_1)
    C_2_ab = np.hypot(a_2, b_2)

    C_bar_ab = (C_1_ab + C_2_ab) / 2
    C_bar_ab_7 = C_bar_ab**7

    G = 0.5 * (1 - np.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25**7)))

    a_p_1 = (1 + G) * a_1
    a_p_2 = (1 + G) * a_2

    C_p_1 = np.hypot(a_p_1, b_1)
    C_p_2 = np.hypot(a_p_2, b_2)

    h_p_1 = np.where(
        np.logical_and(b_1 == 0, a_p_1 == 0),
        0,
        np.degrees(np.arctan2(b_1, a_p_1)) % 360,
    )
    h_p_2 = np.where(
        np.logical_and(b_2 == 0, a_p_2 == 0),
        0,
        np.degrees(np.arctan2(b_2, a_p_2)) % 360,
    )

    delta_C_p = C_p_2 - C_p_1

    h_p_2_s_1 = h_p_2 - h_p_1
    C_p_1_m_2 = C_p_1 * C_p_2
    delta_h_p = np.select(
        [
            C_p_1_m_2 == 0,
            np.fabs(h_p_2_s_1) <= 180,
            h_p_2_s_1 > 180,
            h_p_2_s_1 < -180,
        ],
        [
            0,
            h_p_2_s_1,
            h_p_2_s_1 - 360,
            h_p_2_s_1 + 360,
        ],
    )

    delta_H_p = 2 * np.sqrt(C_p_1_m_2) * np.sin(np.deg2rad(delta_h_p / 2))

    C_bar_p = (C_p_1 + C_p_2) / 2

    a_h_p_1_s_2 = np.fabs(h_p_1 - h_p_2)
    h_p_1_a_2 = h_p_1 + h_p_2
    h_bar_p = np.select(
        [
            C_p_1_m_2 == 0,
            a_h_p_1_s_2 <= 180,
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 < 360),
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
        1
        - 0.17 * np.cos(np.deg2rad(h_bar_p - 30))
        + 0.24 * np.cos(np.deg2rad(2 * h_bar_p))
        + 0.32 * np.cos(np.deg2rad(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.deg2rad(4 * h_bar_p - 63))
    )

    delta_theta = 30 * np.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p**7
    R_C = 2 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + 25**7))

    S_C = 1 + 0.045 * C_bar_p

    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -np.sin(np.deg2rad(2 * delta_theta)) * R_C

    d_Eab = np.sqrt(
        + (delta_C_p / (k_C * S_C)) ** 2
        + (delta_H_p / (k_H * S_H)) ** 2
        + R_T * (delta_C_p / (k_C * S_C)) * (delta_H_p / (k_H * S_H))
    )

    return as_float(d_Eab)
