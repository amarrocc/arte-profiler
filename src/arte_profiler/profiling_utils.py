import logging
import logging.handlers
import subprocess
from pathlib import Path
import pandas as pd
import platform
import importlib.resources
from typing import List, Optional

_loggers = {}  # Dictionary to store loggers


def generate_logger(output_folder: Path, name: str = "profiling"):
    """
    Initializes and configures a logger that writes to both the console and a file.

    Parameters
    ----------
    output_folder : Path
        Directory where the log file will be stored.
    name : str, optional
        Name of the logger, by default "profiling".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    logging.Logger
        Separate logger instance for command execution logs.
    """
    if name in _loggers:
        return _loggers[name], _loggers[f"{name}.command"]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    log_file = output_folder / "profiling.log"

    # File handler (writes logs to a file)
    logfile_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=30 * 1024 * 1024, backupCount=5
    )
    logfile_handler.setLevel(logging.DEBUG)

    # Console handler (writes logs to the terminal)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s - [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logfile_handler, stream_handler],
    )

    # Suppress excessive logs from dependencies
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Create and return a logger instance
    logger = logging.getLogger(name)
    command_logger = logging.getLogger(f"{name}.command")
    command_logger.setLevel(logging.DEBUG)
    command_logger.addHandler(logfile_handler)
    command_logger.propagate = False

    logger.info(f"Logging initialized. Log file: {log_file}")

    # Store logger instances to avoid re-initialization
    _loggers[name] = logger
    _loggers[f"{name}.command"] = command_logger

    return logger, command_logger


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
    """Run a command, optionally redirecting stdin/stdout to files,
    and log stdout/stderr lines as they arrive."""

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
