import argparse
import logging
import importlib.resources
from pathlib import Path
import time
from datetime import datetime
import yaml
import textwrap
import re
import operator as op

import cv2
from . import profiling_utils
import shapely.geometry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v3 as iio
import seaborn as sns
import colour
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from typing import Union, Optional, List
import os
import tempfile


# Register fonts
pdfmetrics.registerFont(
    TTFont(
        "DejaVuSans",
        str(
            importlib.resources.files("arte_profiler")
            / "tools"
            / "dejavu-sans_font"
            / "DejaVuSans.ttf"
        ),
    )
)
pdfmetrics.registerFont(
    TTFont(
        "DejaVuSans-Bold",
        str(
            importlib.resources.files("arte_profiler")
            / "tools"
            / "dejavu-sans_font"
            / "DejaVuSans-Bold.ttf"
        ),
    )
)

# Base path for reference data
TARGETS_BASE_PATH = importlib.resources.files("arte_profiler") / "data" / "targets"
# PROFILES_BASE_PATH = importlib.resources.files("arte_profiler") / "data" / "profiles"
GUIDELINES_BASE_PATH = importlib.resources.files("arte_profiler") / "data" / "guidelines"

OPS = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge}

class BaseColorManager:
    """
    Base class for color chart management, including fiducial detection and RGB extraction.

    Handles loading chart metadata, reference data, and provides methods for
    fiducial detection and RGB value extraction from color chart images.
    """

    def __init__(
        self,
        chart_tif: Union[str, Path],
        chart_type: str = "ColorCheckerSG",
        chart_cie: Optional[Union[str, Path]] = None,
        folder: Optional[Union[str, Path]] = None,
        logger_name: Optional[str] = None,
    ):
        """
        Initialize the BaseColorManager.

        Parameters
        ----------
        chart_tif : str or Path
            Path to the chart image file.
        chart_type : str, optional
            Type of the color chart (default: "ColorCheckerSG").
        chart_cie : str or Path, optional
            Path to the .cie file with Lab reference values.
        folder : str or Path, optional
            Output folder for results and logs.
        logger_name : str, optional
            Name for the logger instance (default: None).
        """
        self.chart_tif = Path(chart_tif)
        if folder is None:
            folder = "."
        self.folder = Path(folder)
        self.chart_type = chart_type
        self.logger, self.command_logger = profiling_utils.generate_logger(
            self.folder, name=logger_name
        )
        self.argyll_bin_path = profiling_utils.get_argyll_bin_path()

        with open(TARGETS_BASE_PATH / "targets_manifest.yaml", "r") as f:
            targets = yaml.safe_load(f)
        if self.chart_type in targets:
            self.reference_data = targets[self.chart_type]
        else:
            raise ValueError(
                f"The specified chart type '{self.chart_type}' is not defined in the targets manifest. "
                f"Available targets are: {', '.join(targets.keys())}."
            )

        self.chart_cht = TARGETS_BASE_PATH / self.reference_data["chart_cht"]
        if chart_cie is None:
            self.chart_cie = TARGETS_BASE_PATH / self.reference_data["chart_cie"]
        else:
            self.chart_cie = Path(chart_cie)

        # Check if all files exist
        for file_path in [
            self.chart_tif,
            self.chart_cht,
            self.chart_cie,
        ]:
            if not file_path.is_file():
                raise FileNotFoundError(f"File {file_path} not found.")

        # Create the directory and all parent directories if they don't exist
        self.folder.mkdir(parents=True, exist_ok=True)

    def find_fiducial(self, max_dim: int = 5000) -> np.ndarray:
        """
        Auto-recognize fiducial marks in the color chart using SIFT.

        Parameters
        ----------
        max_dim : int, optional
            Maximum allowed image dimension for SIFT processing. If the image exceeds this,
            it will be downscaled to improve processing speed. Default is 5000.

        Returns
        -------
        numpy.ndarray
            Array of detected fiducial marks' coordinates.

        Raises
        ------
        RuntimeError
            If fiducial detection fails.
        """
        try:
            self.logger.info(
                f"Setting up fiducial marks detection on {self.chart_tif}..."
            )

            sift = cv2.SIFT_create()
            # Use imageio to read the reference image, use only green channel
            reference = iio.imread(TARGETS_BASE_PATH / self.reference_data["image_path"])[..., 1]
            fiducial_ref = np.array(self.reference_data["fiducial"])
            kp1, ds1 = sift.detectAndCompute(reference, None)

            img2 = iio.imread(self.chart_tif)[..., 1]  # green channel only

            # Check pixel dimensions and scale down if necessary
            scale_factor = 1
            if max(img2.shape[1], img2.shape[0]) > max_dim:
                scale_factor = max_dim / max(img2.shape[1], img2.shape[0])
                self.logger.info(
                    f"Scaling image down by factor {scale_factor:.2f} for faster processing..."
                )
                img2 = cv2.resize(
                    img2,
                    (int(img2.shape[1] * scale_factor), int(img2.shape[0] * scale_factor)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            self.logger.info(f"Determining the fiducial marks...")
            if img2.dtype == np.uint16:
                img2 = ((img2 / 65535) * 255).astype(np.uint8)
            #TODO: check if this works with 8-bit images jpg/tiff?
            kp2, ds2 = sift.detectAndCompute(img2, None)

            if len(kp2) < 4:
                raise RuntimeError("Insufficient keypoints detected in target image.")

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(ds1, ds2, k=2)

            if not matches:
                raise RuntimeError(
                    "No matches found between reference and target image."
                )

            # store all the good matches as per Lowe's ratio test.
            good_m = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_m) < 4:
                raise RuntimeError("Not enough good matches to compute homography.")

            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_m]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_m]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

            if M is None:
                raise RuntimeError("Failed to compute homography.")

            # Transform fiducial reference points
            fiducial_ref_h = np.concatenate(((fiducial_ref), np.ones((4, 1))), axis=-1)
            fiducial_tr = ((M @ fiducial_ref_h.T).T[:, :2]) / (
                (M @ fiducial_ref_h.T).T[:, 2]
            )[..., None]

            # Ensure fiducials are within image bounds
            in_bounds = np.all(
                [
                    fiducial_tr[:, 0] > 0,
                    fiducial_tr[:, 0] < img2.shape[1],
                    fiducial_tr[:, 1] > 0,
                    fiducial_tr[:, 1] < img2.shape[0],
                ]
            )

            # Ensure fiducials form a valid quadrilateral
            poly = shapely.geometry.Polygon(fiducial_tr)
            is_valid_convex_quadrilateral = (
                poly.is_valid and len(poly.convex_hull.exterior.coords) == 5
            )

            if in_bounds and is_valid_convex_quadrilateral:
                fiducial = fiducial_tr * (1 / scale_factor)
                self.logger.info("Fiducial marks successfully detected.")
                return fiducial
            else:
                raise RuntimeError(
                    "Detected fiducials are out of image bounds or invalid."
                )
        except Exception as e:
            self.logger.error(f"Fiducial detection failed: {e}")
            raise RuntimeError("Fiducial auto-recognition failed.") from e

    def extract_rgb_values(
        self,
        fiducial: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Extract RGB values from the color chart image using ArgyllCMS' scanin.

        Parameters
        ----------
        fiducial : list[float], optional
            Coordinates of fiducial marks. If None, auto-detection will be attempted by scanin.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing RGB values and metadata for each patch.

        Raises
        ------
        RuntimeError
            If scanin command fails or RGB extraction is unsuccessful.
        """
        scanin_path = os.path.join(self.argyll_bin_path, "scanin")
        self.logger.debug(f"scanin_path is {scanin_path}")
        scanin_cmd = [
            scanin_path,
            "-v2",
            "-diapn",
            "-O",
            str(self.folder / self.chart_tif.with_suffix(".ti3").name),
            *([f"-F {','.join(map(str, fiducial))}"] if fiducial else []),
            str(self.chart_tif),
            str(self.chart_cht),
            str(self.chart_cie),
            str(self.folder / f"diag_{self.chart_type}.tiff"),
        ]

        self.logger.info(
            f"Running scanin to extract the RGB values of the patches from {self.chart_tif}..."
        )
        retcode = profiling_utils.run_command(
            scanin_cmd, self.command_logger
        )
        if retcode != 0:
            self.logger.error(f"scanin command failed with exit code {retcode}")
            raise RuntimeError("scanin command failed. See logs for details.")

        # Convert the output to DataFrame
        self.chart_ti3 = self.folder / self.chart_tif.with_suffix(".ti3").name
        self.df = profiling_utils.ti3_to_dataframe(str(self.chart_ti3))

        if (
            len(self.df.query("STDEV_R>5"))
            + len(self.df.query("STDEV_G>5"))
            + len(self.df.query("STDEV_B>5"))
            != 0
        ):
            self.logger.warning(
                "Standard deviation of extracted RGB values is > 5!"
            )  # TODO: automate shrink box in case this happens

        return self.df

    def detect_and_extract(self, fiducial: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Detect fiducial marks (if not provided) and extract RGB values from the chart image.

        Parameters
        ----------
        fiducial : list[float], optional
            Coordinates of fiducial marks. If None, auto-detection is performed.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing RGB values and metadata for each patch.
        """
        if fiducial == None:
            fiducial = list(self.find_fiducial().flatten())
        return self.extract_rgb_values(fiducial=fiducial)

    def get_gray_patches(self) -> List[str]:
        """
        Return the index labels of the gray patches in self.df.

        Returns
        -------
        List[str]
            Index labels in self.df corresponding to the gray patches.

        Raises
        ------
        RuntimeError
            If extract_rgb_values() has not been called yet.
        """
        if not hasattr(self, "df"):
            raise RuntimeError(
                "You must call extract_rgb_values() before requesting patches."
            )
        gray_patch_names = self.reference_data["gray_patches"]
        indices = []
        for patch in gray_patch_names:
            col = patch[0]
            row = int(patch[1:])

            match = self.df[(self.df["col"] == col) & (self.df["row"] == row)]
            if match.empty:
                self.logger.warning(f"Gray patch {patch} not found in extracted data.")
            else:
                indices.append(match.index[0])
        return indices

class ProfileCreator(BaseColorManager):
    """
    Class for creating ICC profiles from color chart images.

    Inherits from BaseColorManager and adds methods for generating ICC profiles
    using ArgyllCMS.
    """

    def __init__(
        self,
        chart_tif: Union[str, Path],
        chart_type: str,
        chart_cie: Optional[Union[str, Path]] = None,
        folder: Optional[Union[str, Path]] = None,
        logger_name: Optional[str] = None,
    ):
        """
        Initialize the ProfileCreator.

        Parameters
        ----------
        chart_tif : str or Path
            Path to the chart image file.
        chart_type : str
            Type of the color chart.
        chart_cie : str or Path, optional
            Path to the .cie file with Lab reference values.
        folder : str or Path, optional
            Output folder for results and logs.
        logger_name : str, optional
            Name for the logger instance (default: None).
        """
        super().__init__(chart_tif, chart_type, chart_cie, folder, logger_name=logger_name)

    def icc_from_ti3(self, profile_name: str = "input_profile.icc") -> Path:
        """
        Generate an input ICC profile from the extracted RGB values using 
        ArgyllCMS's colprof.

        Parameters
        ----------
        profile_name : str, optional
            Name for the generated ICC profile file (default: "input_profile.icc").

        Returns
        -------
        pathlib.Path
            Path to the generated ICC profile file.

        Raises
        ------
        RuntimeError
            If RGB values are not extracted before calling this method.
        """
        if not hasattr(self, "chart_ti3"):
            raise RuntimeError(
                "The extraction of the chart's RGB values has not been carried out yet. Please ensure that extract_rgb_values is called first."
            )
        colprof_path = os.path.join(self.argyll_bin_path, "colprof")
        self.logger.debug(f"colprof_path is {colprof_path}")
        colprof_cmd = [
            colprof_path,
            "-v",
            "-Z",
            "a",
            "-ua",
            "-a",
            "g",
            "-b",
            "n",
            "-q",
            "m",
            "-O",
            str(self.folder / profile_name),
            str(self.chart_ti3.with_suffix("")),
        ]

        self.logger.info("Running colprof to build an input ICC profile...")
        retcode = profiling_utils.run_command(colprof_cmd, self.command_logger)
        if retcode != 0:
            self.logger.error(f"colprof command failed with exit code {retcode}")
            raise RuntimeError("colprof command failed. See logs for details.")
        self.in_icc = self.folder / profile_name
        return self.in_icc

    def build_profile(
        self, fiducial: Optional[List[float]] = None, profile_name: str = "input_profile.icc"
    ) -> Path:
        """
        Build an ICC profile from the chart image, including patch extraction and profile generation.

        Parameters
        ----------
        fiducial : list[float], optional
            Coordinates of fiducial marks. If None, auto-detection is performed.
        profile_name : str, optional
            Name for the generated ICC profile file (default: "input_profile.icc").

        Returns
        -------
        pathlib.Path
            Path to the generated ICC profile file.
        """
        self.logger.info(
            f"Profile generation through {self.chart_type} chart initialized."
        )
        self.detect_and_extract(fiducial=fiducial)
        self.icc_from_ti3(profile_name=profile_name)
        self.logger.info(f"Profile generated: {self.in_icc}")
        return self.in_icc


class ProfileEvaluator(BaseColorManager):
    """
    Class for evaluating ICC profiles using color chart images.

    Inherits from BaseColorManager and provides methods for profile evaluation,
    including Delta E computation, visualization, and report generation.
    """

    def __init__(
        self,
        chart_tif: Union[str, Path],
        chart_type: str,
        in_icc: Union[str, Path],
        chart_cie: Optional[Union[str, Path]] = None,
        folder: Optional[Union[str, Path]] = None,
        patch_data: Optional[pd.DataFrame] = None,
        logger_name: Optional[str] = None,
    ):
        """
        Initialize the ProfileEvaluator.

        Parameters
        ----------
        chart_tif : str or Path
            Path to the chart image file.
        chart_type : str
            Type of the color chart.
        in_icc : str or Path
            Path to the ICC profile to evaluate.
        chart_cie : str or Path, optional
            Path to the .cie file with Lab reference values.
        folder : str or Path, optional
            Output folder for results and logs.
        patch_data : pandas.DataFrame, optional
            Pre-extracted patch data (if available).
        logger_name : str, optional
            Name for the logger instance (default: None).
        """
        super().__init__(chart_tif, chart_type, chart_cie, folder, logger_name=logger_name)
        self.in_icc = Path(in_icc)
#        self.out_icc = out_icc
        self.df = patch_data

        with open(GUIDELINES_BASE_PATH / "guidelines.yaml", "r") as f:
            self.guidelines = yaml.safe_load(f)

        # with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
        #     profiles = yaml.safe_load(f)
        # if out_icc in profiles.keys():
        #     self.out_icc = PROFILES_BASE_PATH / profiles[out_icc]["path"]
        # else:
        #     self.out_icc = Path(out_icc)

        # Check if all files exist
        for file_path in [
            self.in_icc,
            # self.out_icc,
        ]:
            if not file_path.is_file():
                raise FileNotFoundError(f"File {file_path} not found.")
            
    def get_corrected_lab_vals(self) -> np.ndarray:
        """
        Compute corrected Lab values for the color chart patches using the
        input ICC profile.

        Returns
        -------
        numpy.ndarray
            Array of corrected Lab values for the patches.
        """
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as input_file, \
             tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
            self.df.to_csv(input_file.name, sep='\t',
                           columns=['RGB_R', 'RGB_G', 'RGB_B'], header=False, index=False)
            icclu_path = os.path.join(self.argyll_bin_path, "icclu")
            icclu_cmd = [
                icclu_path,
                "-s",
                "100",
                "-p",
                "l",
                "-v",
                "0",
                str(self.in_icc),
            ]

            self.logger.info("Running icclu...")
            retcode = profiling_utils.run_command(
                icclu_cmd,
                self.command_logger,
                stdin_path=input_file.name,
                stdout_path=output_file.name,
            )
            if retcode != 0:
                self.logger.error(f"icclu command failed with exit code {retcode}")
                raise RuntimeError("icclu command failed. See logs for details.")

            corr_lab_vals = np.loadtxt(output_file.name)

        self.df["corr_L"] = corr_lab_vals[:, 0]
        self.df["corr_A"] = corr_lab_vals[:, 1]
        self.df["corr_B"] = corr_lab_vals[:, 2]

        os.remove(input_file.name)
        os.remove(output_file.name)

        return corr_lab_vals

    def get_gt_lab_vals(self) -> np.ndarray:
        """
        Retrieve ground truth Lab values for the color chart patches from the .cie file.

        Returns
        -------
        numpy.ndarray
            Array of ground truth Lab values.
        """
        # Build DataFrame with Lab ground truth values from .cie file
        gt_lab_df = pd.DataFrame(
            profiling_utils.parse_file(self.chart_cie),
            columns=["SAMPLE_ID", "LAB_L", "LAB_A", "LAB_B"],
        )
        for col in gt_lab_df.columns[1:]:
            gt_lab_df[col] = gt_lab_df[col].astype(float)

        gt_lab_vals = (
            gt_lab_df[["LAB_L", "LAB_A", "LAB_B"]]
            .values.reshape(
                (self.reference_data["rows"], self.reference_data["cols"], 3), order="F"
            )
            .reshape((self.reference_data["rows"] * self.reference_data["cols"], 3))
        )

        # self.gt_lab_vals = gt_lab_df[["LAB_L", "LAB_A", "LAB_B"]].values #old scanin

        self.df["gt_L"] = gt_lab_vals[:, 0]
        self.df["gt_A"] = gt_lab_vals[:, 1]
        self.df["gt_B"] = gt_lab_vals[:, 2]

        return gt_lab_vals

    def get_guideline_level_passed(
        self,
        guideline: str,
        param: str,
        value: float,
        object_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return the first (strictest) level in guidelines.yaml that 'value' satisfies.

        Parameters
        ----------
        guideline : str
            The name of the guideline (e.g., "FADGI" or "Metamorfoze").
        param : str
            The metric/parameter to check (e.g., "delta_e_mean", "delta_e_max").
        value : float
            The value to be evaluated against the guideline thresholds.
        object_type : str, optional
            The object type for FADGI (e.g., "paintings_2d"). Required for FADGI, ignored for Metamorfoze.

        Returns
        -------
        str or None
            The first (strictest) level passed (e.g., "4_star", "metamorfoze"), or None if none are passed.

        Raises
        ------
        KeyError
            If the guideline, object_type, or param is not found in the guidelines.
        """
        # ── guideline layer ────────────────────────────────────────────────
        try:
            guide_block = self.guidelines[guideline]
        except KeyError:
            raise KeyError(f"Unknown guideline '{guideline}'")

        # ── object-type layer (only for FADGI) ─────────────────────────────
        if guideline == "FADGI":
            if object_type is None:
                raise KeyError("FADGI requires an 'object_type'")
            try:
                levels = guide_block[object_type]
            except KeyError:
                raise KeyError(
                    f"Unknown object_type '{object_type}' for guideline 'FADGI'"
                )
        else:
            if object_type is not None:
                self.logger.warning(
                    f"'object_type' ignored for guideline '{guideline}'"
                )
            levels = guide_block

        # ── metric evaluation ──────────────────────────────────────────────
        metric_defined = False
        for level, rules in levels.items():       # YAML order preserved
            if param not in rules:
                continue
            metric_defined = True
            rule = rules[param]
            if OPS[rule["operator"]](value, rule["value"]):
                return level                      # first pass ⇒ strictest

        if not metric_defined:
            raise KeyError(
                f"Metric '{param}' not defined for {guideline}"
                + (f'/{object_type}' if object_type else '')
            )
        return None  

    def compute_delta_e(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Delta E values (CIE 1976 and CIE 2000) for corrected vs. 
        reference Lab values.

        Returns
        -------
        tuple of np.ndarray
            (DeltaE 1976 values, DeltaE 2000 values) for each patch.
        """
        self.logger.info(
            "Reading the corrected and ground truth values of the patches..."
        )
        if "gt_L" not in self.df.columns:
            gt_lab_vals = self.get_gt_lab_vals()
        else:
            gt_lab_vals = self.df[["gt_L", "gt_A", "gt_B"]].values
        if "corr_L" not in self.df.columns:
            corr_lab_vals = self.get_corrected_lab_vals()
        else:
            corr_lab_vals = self.df[["corr_L", "corr_A", "corr_B"]].values

        self.logger.info("Computing DeltaE...")
        delta_e_76 = colour.difference.delta_E_CIE1976(gt_lab_vals, corr_lab_vals)
        delta_e_2000 = colour.difference.delta_E_CIE2000(gt_lab_vals, corr_lab_vals)

        self.df["delta_e_76"] = delta_e_76
        self.df["delta_e_2000"] = delta_e_2000

        de_76_mean = delta_e_76.mean()
        de_76_max = delta_e_76.max()
        de_2000_mean = delta_e_2000.mean()
        de_2000_quantile = np.quantile(delta_e_2000, 0.90)

        self.logger.info(f"ΔE*76 mean: {de_76_mean:.2f}, max: {de_76_max:.2f}")
        self.logger.info(
            f"ΔE*2000 mean: {de_2000_mean:.2f}, 90th_percentile: {de_2000_quantile:.2f}"
        )

        # Metamorfoze compliance
        meta_mean_level = self.get_guideline_level_passed("Metamorfoze", "delta_e_mean", de_76_mean)
        meta_max_level = self.get_guideline_level_passed("Metamorfoze", "delta_e_max", de_76_max)
        if meta_mean_level == "metamorfoze" and meta_max_level == "metamorfoze":
            pass  # compliant, no warning
        else:
            self.logger.warning(
                "Color accuracy is not compliant with Metamorfoze (metamorfoze quality level) guidelines!"
            )

        # FADGI compliance (4-star)
        fadgi_mean_level = self.get_guideline_level_passed("FADGI", "delta_e_mean", de_2000_mean, "paintings_2d")
        fadgi_90th_level = self.get_guideline_level_passed("FADGI", "delta_e_90th_percentile", de_2000_quantile, "paintings_2d")
        if fadgi_mean_level == "4_star" and fadgi_90th_level == "4_star":
            pass  # compliant, no warning
        else:
            self.logger.warning(
                "Color accuracy is not compliant with FADGI (4-star) guidelines!"
            )

        return delta_e_76, delta_e_2000

    def compute_oecf(self) -> np.ndarray:
        """
        Compute OECF (ΔL*2000) for the gray patches in the chart.

        Returns
        -------
        numpy.ndarray
            Array of ΔL*2000 values (one per gray patch).
        """
        gray_idx = self.get_gray_patches()
        # Ensure Lab values are present
        if "gt_L" not in self.df.columns:
            self.get_gt_lab_vals()
        if "corr_L" not in self.df.columns:
            self.get_corrected_lab_vals()

        delta_L2000 = profiling_utils.delta_L_CIE2000(
            self.df.loc[gray_idx, "corr_L"].values,
            self.df.loc[gray_idx, "gt_L"].values
        )
        self.df["oecf"] = np.nan
        self.df.loc[gray_idx, "oecf"] = delta_L2000
        
        self.logger.info(f"OECF (ΔL*2000) mean: {delta_L2000.mean():.2f}, max: {delta_L2000.max():.2f}")
        # FADGI compliance (4-star)
        level = self.get_guideline_level_passed("FADGI", "oecf", delta_L2000.max(), "paintings_2d")
        if level != "4_star":
            self.logger.warning(
                "OECF (ΔL*2000) is not compliant with FADGI (4-star) guidelines!"
            )

        return delta_L2000

    def compute_white_balance(self) -> np.ndarray:
        """
        Compute the white balance error (ΔE(a*b*)) for the gray patches in the chart.

        Returns
        -------
        numpy.ndarray
            Array of ΔE(a*b*) values (one per gray patch).
        """
        gray_idx = self.get_gray_patches()
        if "gt_L" not in self.df.columns:
            self.get_gt_lab_vals()
        if "corr_L" not in self.df.columns:
            self.get_corrected_lab_vals()

        delta_Eab2000 = profiling_utils.delta_Eab_CIE2000(
            self.df.loc[gray_idx, ["corr_A", "corr_B"]].values, 
            self.df.loc[gray_idx, ["gt_A", "gt_B"]].values
        )
        self.df["white_balance"] = np.nan
        self.df.loc[gray_idx, "white_balance"] = delta_Eab2000

        self.logger.info(f"White balance (ΔE(a*b*)) mean: {delta_Eab2000.mean():.2f}, max: {delta_Eab2000.max():.2f}")
        # FADGI compliance (4-star)
        level = self.get_guideline_level_passed("FADGI", "white_balance", delta_Eab2000.max(), "paintings_2d")
        if level != "4_star":
            self.logger.warning(
                "White balance (ΔE(a*b*)) is not compliant with FADGI (4-star) guidelines!"
            )

        return delta_Eab2000

    def _plot_patch_chart_with_text(
        self, text_func, filename: str, title: str
    ) -> Path:
        """
        Helper to plot a patch chart with custom text overlay. 
        
        Generates an sRGB color chart comparing corrected and reference colors 
        for each patch, annotates it with custom values, and saves it to the 
        output folder.

        Parameters
        ----------
        text_func : callable
            Function (row, col) -> str or None. Returns text for each 
            patch or None for no text.
        filename : str
            Output filename (relative to self.folder).
        title : str
            Title for the plot.

        Returns
        -------
        pathlib.Path
            Path to the saved image.
        """
        # convert corrected Lab values to sRGB (illuminant D50)
        corr_sRGB = colour.XYZ_to_sRGB(
            colour.Lab_to_XYZ(
                self.df[["corr_L", "corr_A", "corr_B"]].values,
                illuminant=np.array([0.3457, 0.3585]),
            ),
            illuminant=np.array([0.3457, 0.3585]),
        ).clip(0, 1)
        corr_sRGB = (corr_sRGB * 255).astype("uint8")

        # convert reference Lab values to sRGB (illuminant D50)
        sRGB = colour.XYZ_to_sRGB(
            colour.Lab_to_XYZ(
                self.df[["gt_L", "gt_A", "gt_B"]].values,
                illuminant=np.array([0.3457, 0.3585]),
            ),
            illuminant=np.array([0.3457, 0.3585]),
        ).clip(0, 1)
        sRGB = (sRGB * 255).astype("uint8")

        # Visualize sRGB reference and corrected (through input profile) colors
        self.delta_e_size = (1400, 1000)
        dpi = 100
        fig1, ax1 = plt.subplots(
            figsize=(self.delta_e_size[0] / dpi, self.delta_e_size[1] / dpi)
        )

        size = 100
        spacing = 20

        img = (
            np.ones(
                (
                    size * self.reference_data["rows"]
                    + (self.reference_data["rows"] + 1) * spacing,
                    size * self.reference_data["cols"]
                    + (self.reference_data["cols"] + 1) * spacing,
                    3,
                )
            )
            * (50, 50, 50)
        ).astype(np.uint16)

        index = 0
        for row in np.arange(self.reference_data["rows"]):
            for col in np.arange(self.reference_data["cols"]):
                # Calculate rectangle positions
                x1 = spacing + spacing * col + size * col
                y1 = spacing + spacing * row + size * row
                x2 = x1 + size // 2
                y2 = y1 + size

                # Draw ground truth rectangle
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (
                        sRGB[..., 0][index].item(),
                        sRGB[..., 1][index].item(),
                        sRGB[..., 2][index].item(),
                    ),
                    -1,
                )

                # Draw corrected rectangle
                cv2.rectangle(
                    img,
                    (x2, y1),
                    (x1 + size, y2),
                    (
                        corr_sRGB[..., 0][index].item(),
                        corr_sRGB[..., 1][index].item(),
                        corr_sRGB[..., 2][index].item(),
                    ),
                    -1,
                )

                # Calculate text position
                text_x = x1 + size / 2  # Center x
                text_y = (y1 + y2) / 2  # Center y

                text = text_func(row, col)
                if text is not None:
                    ax1.text(
                        text_x,
                        text_y,
                        text,
                        fontsize=16,
                        color="white",
                        va="center",
                        ha="center",
                        backgroundcolor=(0.4, 0.4, 0.4, 0.4),
                    )
                index += 1

        ax1.imshow(img)
        ax1.set_xticks(
            np.arange(
                (spacing + size / 2),
                self.reference_data["cols"] * (spacing + size) + spacing,
                spacing + size,
            )
        )
        ax1.set_xticklabels(self.df.col.unique(), fontsize=16)

        ax1.set_yticks(
            np.arange(
                (spacing + size / 2),
                self.reference_data["rows"] * (spacing + size) + spacing,
                spacing + size,
            )
        )
        ax1.set_yticklabels(self.df.row.unique(), fontsize=16)
        plt.title(title, fontsize=16)
        fig1.tight_layout()
        fig1.savefig(self.folder / filename, facecolor="w", dpi=dpi)
        plt.close(fig1)
        return self.folder / filename

    def create_de_patch_chart(self) -> Path:
        """
        Visualize ΔE₀₀ values for all patches on the patch chart.

        Returns
        -------
        pathlib.Path
            Path to the saved delta E patch chart image.
        """
        if "delta_e_2000" not in self.df.columns:
            self.logger.error("Delta E 2000 values not found. Run compute_delta_e() before plotting.")
            raise RuntimeError("Delta E 2000 values not found. Run compute_delta_e() before plotting.")
        def text_func(row, col):
            val = self.df["delta_e_2000"].values.reshape(
                self.reference_data["rows"], self.reference_data["cols"]
            )[row, col]
            text = str(round(val, 2))
            if len(text) < 4:
                text = text + "0"
            return text

        return self._plot_patch_chart_with_text(
            text_func,
            f"delta_e_{self.chart_type}.png",
            r"$\Delta{{E}}_{{00}}^{{*}}$ for the patches"
        )

    def create_de_histogram(self) -> Path:
        """
        Create and save a histogram of the ΔE₀₀ values for the chart patches.

        Returns
        -------
        pathlib.Path
            Path to the saved histogram image.
        """
        if "delta_e_2000" not in self.df.columns:
            self.logger.error("Delta E 2000 values not found. Run compute_delta_e() before plotting.")
            raise RuntimeError("Delta E 2000 values not found. Run compute_delta_e() before plotting.")
        self.delta_e_hist_size = (1400, 1000)
        dpi = 100
        fig = plt.figure(
            figsize=(self.delta_e_hist_size[0] / dpi, self.delta_e_hist_size[1] / dpi)
        )
        ax = fig.add_subplot(111)
        ax.hist(self.df["delta_e_2000"], bins=20, range=(0, 4))
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)
        ax.text(
            0.7,
            0.95,
            f"$\Delta{{E}}_{{00}}^{{*}}$ mean: {self.df['delta_e_2000'].mean():.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ 90%: {np.quantile(self.df['delta_e_2000'], 0.90):.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ max: {self.df['delta_e_2000'].max():.2f}",
            transform=plt.gca().transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props,
        )
        ax.set_xlabel(f"$\Delta{{E}}_{{00}}^{{*}}$", fontsize=16)
        ax.set_ylabel("Number of patches", fontsize=16)

        fig.tight_layout()
        fig.savefig(self.folder / f"delta_e_hist_{self.chart_type}.png", facecolor="w", dpi=dpi)
        plt.close(fig)

        return self.folder / f"delta_e_hist_{self.chart_type}.png"

    def create_oecf_patch_chart(self) -> Path:
        """
        Visualize OECF (ΔL*2000) values for gray patches on the patch chart.

        Only gray patches will have text (OECF value), others will be blank.

        Returns
        -------
        pathlib.Path
            Path to the saved OECF patch chart image.
        """
        if "oecf" not in self.df.columns:
            self.logger.error("oecf values not found. Run compute_oecf() before plotting.")
            raise RuntimeError("oecf values not found. Run compute_oecf() before plotting.")
        def text_func(row, col):
            val = self.df["oecf"].values.reshape(
                self.reference_data["rows"], self.reference_data["cols"]
            )[row, col]
            if np.isnan(val):
                return None
            else:
                text = str(round(val, 2))
                if len(text) < 4:
                    text = text + "0"
                return text

        return self._plot_patch_chart_with_text(
            text_func,
            f"oecf_{self.chart_type}.png",
            r"OECF ($\Delta L^*_{00}$) for gray patches"
        )

    def create_white_balance_patch_chart(self) -> Path:
        """
        Visualize white balance (ΔE(a*b*)) values for gray patches on the patch chart.

        Only gray patches will have text (white balance value), others will be blank.

        Returns
        -------
        pathlib.Path
            Path to the saved white balance patch chart image.
        """
        if "white_balance" not in self.df.columns:
            self.logger.error("white_balance values not found. Run compute_white_balance() before plotting.")
            raise RuntimeError("white_balance values not found. Run compute_white_balance() before plotting.")
        def text_func(row, col):
            val = self.df["white_balance"].values.reshape(
                self.reference_data["rows"], self.reference_data["cols"]
            )[row, col]
            if np.isnan(val):
                return None
            else:
                text = str(round(val, 2))
                if len(text) < 4:
                    text = text + "0"
                return text

        return self._plot_patch_chart_with_text(
            text_func,
            f"white_balance_{self.chart_type}.png",
            r"White balance ($\Delta{{E}}_{{00}}^{{*}}({a}^{*}{b}^{*})$) for gray patches"
        )

    def plot_stdev_patches(self) -> Path:
        """
        Generate heatmaps of the standard deviation of RGB values for the chart patches.

        This method creates heatmaps for the R, G, and B standard deviation values of the extracted patches
        and saves them to the output folder.

        Returns
        -------
        pathlib.Path
            Path to the saved standard deviation heatmap image.
        """
        dpi = 100
        self.stdev_patches_size = (1000, 2000)
        fig, ax = plt.subplots(
            3,
            1,
            figsize=(
                self.stdev_patches_size[0] / dpi,
                self.stdev_patches_size[1] / dpi,
            ),
        )
        sns.heatmap(
            self.df[["col", "row", "STDEV_R"]].pivot(
                index="row", columns="col", values="STDEV_R"
            ),
            annot=True,
            linewidths=0.5,
            ax=ax[0],
            cmap="mako",
            vmin=0.1,
            vmax=2,
        )
        sns.heatmap(
            self.df[["col", "row", "STDEV_G"]].pivot(
                index="row", columns="col", values="STDEV_G"
            ),
            annot=True,
            linewidths=0.5,
            ax=ax[1],
            cmap="mako",
            vmin=0.1,
            vmax=2,
        )
        sns.heatmap(
            self.df[["col", "row", "STDEV_B"]].pivot(
                index="row", columns="col", values="STDEV_B"
            ),
            annot=True,
            linewidths=0.5,
            ax=ax[2],
            cmap="mako",
            vmin=0.1,
            vmax=2,
        )

        ax[0].set_title("STDEV_R")
        ax[1].set_title("STDEV_G")
        ax[2].set_title("STDEV_B")

        fig.tight_layout()
        fig.savefig(self.folder / f"stdev_patches_{self.chart_type}.png", facecolor="w", dpi=dpi)
        plt.close(fig)
        return self.folder / f"stdev_patches_{self.chart_type}.png"

    def generate_report(
        self, title: str = "Profiling Report", filename: str = "profiling_report.pdf"
    ) -> Path:
        """
        Generate a PDF report summarizing the analysis results.

        The report includes:
        - ΔE color comparison charts.
        - A histogram of ΔE values.
        - Heatmaps of the standard deviation of RGB values.
        - Metadata and conclusions based on FADGI and Metamorfoze guidelines.

        Parameters
        ----------
        title : str, optional
            Title for the PDF report (default: "Profiling Report").
        filename : str, optional
            Name of the PDF report file (default: "profiling_report.pdf").

        Returns
        -------
        pathlib.Path
            Path to the generated PDF report.
        """
        self.logger.info("Generating report...")
        c = canvas.Canvas(str(self.folder / filename), pagesize=A4)

        canvas_width, canvas_height = A4

        # title
        c.setFont("DejaVuSans-Bold", 12)
        c.drawString(100, 800, title)
        c.setFont("DejaVuSans", 11)
        t = datetime.fromtimestamp(time.time())
        c.drawString(100, 780, f"Generated: {t.date()} at {str(t.time())[:-7]}")
        c.drawString(100, 760, f"Using: {self.chart_type} chart in image {self.chart_tif.name}")
        c.drawString(100, 740, f"Profile: {self.in_icc.name}")
 
        # Color accuracy
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 700, f"Color accuracy")

        de_chart_path = self.create_de_patch_chart()
        de_chart_hist_path = self.create_de_histogram()

        c.drawImage(
            de_chart_path,
            100,
            400,
            width=self.delta_e_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )
        c.drawImage(
            de_chart_hist_path,
            100,
            100,
            width=self.delta_e_hist_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )

        de_76_mean = self.df["delta_e_76"].mean()
        de_76_max = self.df["delta_e_76"].max()
        de_2000_mean = self.df["delta_e_2000"].mean()
        de_2000_quantile = np.quantile(self.df["delta_e_2000"], 0.90)

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 80, f"ΔE* mean: {de_76_mean:.2f}, ΔE* max: {de_76_max:.2f}")
        c.drawString(
            100,
            60,
            f"ΔE₀₀* mean: {de_2000_mean:.2f}, ΔE₀₀* 90%: {de_2000_quantile:.2f}",
        )

        meta_mean_level = self.get_guideline_level_passed("Metamorfoze", "delta_e_mean", de_76_mean)
        meta_max_level = self.get_guideline_level_passed("Metamorfoze", "delta_e_max", de_76_max)
        # Determine lowest (least strict) level passed, or None if either is None
        meta_levels = list(self.guidelines["Metamorfoze"].keys())
        def _level_index(level):
            return meta_levels.index(level) if level in meta_levels else len(meta_levels)
        if meta_mean_level is None or meta_max_level is None:
            meta_level_passed = None
        else:
            meta_level_passed = meta_mean_level if _level_index(meta_mean_level) > _level_index(meta_max_level) else meta_max_level

        fadgi_mean_level = self.get_guideline_level_passed("FADGI", "delta_e_mean", de_2000_mean, "paintings_2d")
        fadgi_90th_level = self.get_guideline_level_passed("FADGI", "delta_e_90th_percentile", de_2000_quantile, "paintings_2d")
        fadgi_levels = list(self.guidelines["FADGI"]["paintings_2d"].keys())
        def _fadgi_level_index(level):
            return fadgi_levels.index(level) if level in fadgi_levels else len(fadgi_levels)
        if fadgi_mean_level is None or fadgi_90th_level is None:
            fadgi_level_passed = None
        else:
            fadgi_level_passed = fadgi_mean_level if _fadgi_level_index(fadgi_mean_level) > _fadgi_level_index(fadgi_90th_level) else fadgi_90th_level

        if meta_level_passed == meta_levels[0]:
            c.setFillColor("green")
        elif meta_level_passed is None:
            c.setFillColor("red")
        else:
            c.setFillColor("black")
        c.drawString(320, 80, f"Metamorfoze: {meta_level_passed if meta_level_passed else 'no level passed'}")

        if fadgi_level_passed == fadgi_levels[0]:
            c.setFillColor("green")
        elif fadgi_level_passed is None:
            c.setFillColor("red")
        else:
            c.setFillColor("black")
        c.drawString(320, 60, f"FADGI: {fadgi_level_passed if fadgi_level_passed else 'no level passed'} (Paintings and Other 2D Art)")

        c.setFillColor("black")
        c.showPage()

        # OECF
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 800, f"OECF")

        oecf_chart_path = self.create_oecf_patch_chart()
        c.drawImage(
            oecf_chart_path,
            100,
            500,
            width=self.delta_e_size[0] // 3.5,
            height=self.delta_e_size[1] // 3.5,
        )

        oecf_vals = self.df["oecf"].dropna()
        if not oecf_vals.empty:
            oecf_mean = oecf_vals.mean()
            oecf_max = oecf_vals.max()
            c.setFont("DejaVuSans", 11)
            c.drawString(100, 480, f"ΔL*2000 mean: {oecf_mean:.2f}, max: {oecf_max:.2f}")
            fadgi_oecf_level = self.get_guideline_level_passed("FADGI", "oecf", oecf_max, "paintings_2d")
            if fadgi_oecf_level == fadgi_levels[0]:
                c.setFillColor("green")
            elif fadgi_oecf_level is None:
                c.setFillColor("red")
            else:
                c.setFillColor("black")
            c.drawString(320, 480, f"FADGI: {fadgi_oecf_level if fadgi_oecf_level else 'no level passed'} (Paintings and Other 2D Art)")
            c.setFillColor("black")

        # White balance
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 440, f"White balance")
        wb_chart_path = self.create_white_balance_patch_chart()
        c.drawImage(
            wb_chart_path,
            100,
            140,
            width=self.delta_e_size[0] // 3.5,
            height=self.delta_e_size[1] // 3.5,
        )
        wb_vals = self.df["white_balance"].dropna()
        if not wb_vals.empty:
            wb_mean = wb_vals.mean()
            wb_max = wb_vals.max()
            c.setFont("DejaVuSans", 11)
            c.drawString(100, 120, f"ΔE(a*b*) mean: {wb_mean:.2f}, max: {wb_max:.2f}")
            fadgi_wb_level = self.get_guideline_level_passed("FADGI", "white_balance", wb_max, "paintings_2d")
            if fadgi_wb_level == fadgi_levels[0]:
                c.setFillColor("green")
            elif fadgi_wb_level is None:
                c.setFillColor("red")
            else:
                c.setFillColor("black")
            c.drawString(320, 120, f"FADGI: {fadgi_wb_level if fadgi_wb_level else 'no level passed'} (Paintings and Other 2D Art)")
        c.setFillColor("black")
        c.showPage()

        # appendix
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 800, f"Appendix")
        c.setFont("DejaVuSans", 11)
        c.drawString(
            100, 780, f"Standard deviation of the extracted RGB values of the patches"
        )
        stdev_chart = self.plot_stdev_patches()
        c.drawImage(
            stdev_chart,
            100,
            100,
            width=self.stdev_patches_size[0] // 3,
            height=self.stdev_patches_size[1] // 3,
        )
        c.showPage()

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 780, f"Extracted patches")
        diag_tiff_path = self.folder / f"diag_{self.chart_type}.tiff"
        diag_png_path = self.folder / f"diag_{self.chart_type}.png"
        diag = iio.imread(diag_tiff_path)
        if diag.shape[1] < diag.shape[0]:
            diag = np.rot90(diag)
        diag_thumb = cv2.resize(diag, (int(1000*(diag.shape[1]/diag.shape[0])), 1000), interpolation=cv2.INTER_AREA)
        iio.imwrite(diag_png_path, diag_thumb)
        c.drawImage(
            diag_png_path,
            100,
            480,
            width=diag_thumb.shape[1] // 3.5,
            height=diag_thumb.shape[0] // 3.5,
        )

        # Save the PDF
        c.save()
        return self.folder / filename

    def make_plots(self) -> tuple[Path, Path, Path, Path, Path]:
        """
        Generate all plots and visualizations for the evaluation.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]
            Paths to the generated images: (patch comparison chart, delta E histogram, OECF patch chart, white balance patch chart, stdev heatmap).
        """
        de_patch_chart = self.create_de_patch_chart()
        de_hist = self.create_de_histogram()
        oecf_patch_chart = self.create_oecf_patch_chart()
        wb_patch_chart = self.create_white_balance_patch_chart()
        stdev_chart = self.plot_stdev_patches()
        return de_patch_chart, de_hist, oecf_patch_chart, wb_patch_chart, stdev_chart

    def evaluate_profile(
        self,
        fiducial: Optional[List[float]] = None,
        report_title: str = "Profiling Report",
        report_filename: str = "profiling_report.pdf",
    ) -> Path:
        """
        Run the full evaluation pipeline: extract patches, compute Delta E, generate plots, and create a report.

        Parameters
        ----------
        fiducial : list[float], optional
            Coordinates of fiducial marks. If None, auto-detection is performed.
        report_title : str, optional
            Title for the PDF report (default: "Profiling Report").
        report_filename : str, optional
            Name of the PDF report file (default: "profiling_report.pdf").

        Returns
        -------
        pathlib.Path
            Path to the generated PDF report.
        """
        self.logger.info(
            f"Profile evaluation through {self.chart_type} chart initialized."
        )
        if self.df is None:
            self.detect_and_extract(fiducial)
        self.compute_delta_e()
        self.compute_oecf()
        self.compute_white_balance()
        self.generate_report(report_title, report_filename)
        self.logger.info(
            f"Report completed. Results saved in {self.folder}."
        )
        return self.folder / report_filename


def parse_args():
    """
    Parses the command-line arguments for arte-profiler.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    with open(TARGETS_BASE_PATH / "targets_manifest.yaml", "r") as f:
        available_targets = list(yaml.safe_load(f).keys())

    # with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
    #     available_profiles = list(yaml.safe_load(f).keys())

    description = textwrap.dedent("""\
        A Python wrapper around ArgyllCMS for camera profiling and color 
        accuracy evaluation. The program supports:
        
        1. Building a color profile from an input image of a supported chart;
        2. Evaluating an existing ICC profile using an input image of a 
           supported color chart;
        3. Building and evaluating a color profile in a single run, either from 
           an image that contains two different supported charts or from two 
           separate images. In this case, one chart is used to generate 
           the color profile, while the other one assesses its accuracy.  
        
        In all cases, arte-profiler generates a structured PDF report 
        summarizing the results based on the Metamorfoze and FADGI guidelines.
    """)
    
    epilog = textwrap.dedent("""\
        Examples
        --------
        1) Build a profile from a single chart image:
           arte-profiler \\
             --build_tif path/to/chart_image.tiff \\
             --build_type chart_type \\
             --test_tif path/to/chart_image.tiff \\
             --test_type chart_type \\
             -O output_folder
           
           Note: This builds and tests the profile on the same chart. The 
           report mainly confirms correct generation and application, not 
           accuracy. For proper evaluation, use a separate chart (see case 3) 
           to avoid overestimating performance.

        2) Evaluate an existing ICC profile:
           arte-profiler \\
             --test_tif path/to/chart_image.tiff \\
             --test_type chart_type \\
             --in_icc path/to/existing_profile.icc \\
             -O output_folder
        
        3) Generate and evaluate a color profile in a single run:
          arte-profiler \\
            --build_tif path/to/chartA_image.tiff \\
            --build_type chartA_type \\
            --test_tif path/to/chartB_image.tiff \\
            --test_type chartB_type \\
            -O output_folder
                             
          Note: you can point both --build_tif and --test_tif to the same 
          image if it contains two different supported charts, instead of using 
          two separate files.
                             
    """)

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--build_tif",
        help="Path to image containing the chart used to build a new ICC profile. (May contain one or two charts).",
    )

    parser.add_argument(
        "--test_tif",
        help="""Path to image containing the chart used to test an ICC profile. 
            (May contain one or two charts).""",
        required=True,
    )

    parser.add_argument(
        "--build_type",
        choices=available_targets,
        help="Chart type for profile creation. Required if --build_tif is specified.",
    )

    parser.add_argument(
        "--test_type",
        choices=available_targets,
        help="Chart type for evaluation.",
        required=True,
    )

    parser.add_argument(
        "--build_cie",
        type=str,
        help="""Path to the .cie file with Lab reference values for the 
            build chart. If not provided, generic reference values will be 
            used.""",
    )

    parser.add_argument(
        "--test_cie",
        type=str,
        help="""Path to the .cie file with Lab reference values for the 
            test chart. If not provided, generic reference values will be 
            used.""",
    )

    parser.add_argument(
        "--fiducial_build",
        type=str,
        help="""Manually specify fiducial marks for build chart as 
            x1,y1,x2,y2,x3,y3,x4,y4. Overrides auto-detection.""",
    )

    parser.add_argument(
        "--fiducial_test",
        type=str,
        help="""Manually specify fiducial marks for test chart as 
            x1,y1,x2,y2,x3,y3,x4,y4. Overrides auto-detection.""",
    )

    parser.add_argument(
        "--in_icc",
        type=str,
        help="""Existing ICC profile to evaluate. Required when --build_tif is 
            not provided.""",
    )

    # parser.add_argument(
    #     "--out_icc",
    #     help=f"The output ICC profile. Can be one of: {available_profiles} or a specified path.",
    #     required=True,
    # )

    parser.add_argument(
        "-O",
        "--out_folder",
        type=str,
        default=".",
        help="""The output folder where to write output from this program 
            (default: current directory)""",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if not args.build_tif and not args.in_icc:
        parser.error(
            "You must provide an existing profile to test with --in_icc or generate one from an input image with --build_tif and --build_type."
        )
    if args.build_tif and args.in_icc:
        parser.error(
            "You cannot specify --build_tif and --in_icc and at the same time." \
            "Either provide an existing profile to test with --in_icc or generate a new one from an input image with --build_tif and --build_type."
        )
    if (args.build_tif and not args.build_type) or (args.build_type and not args.build_tif):
        parser.error(
            "You must provide both --build_tif and --build_type to specify the build chart image path and type."
        )
    

    return args


def main():
    """
    Main entry point for arte-profiler.

    Parses arguments and runs the appropriate workflow for profile creation and/or evaluation.
    """
    args = parse_args()

    # Convert potential fiducial coords into list of ints if provided
    fiducial_list_build = (
        list(map(int, args.fiducial_build.split(","))) if args.fiducial_build else None
    )
    fiducial_list_test = (
        list(map(int, args.fiducial_test.split(","))) if args.fiducial_test else None
    )

    CLI_LOGGER_NAME = "profiling"

    if args.build_tif and args.build_type:
        # Build a color profile
        creator = ProfileCreator(
            chart_tif=args.build_tif,
            chart_type=args.build_type,
            chart_cie=args.build_cie,
            folder=args.out_folder,
            logger_name=CLI_LOGGER_NAME,
        )
        creator.build_profile(fiducial_list_build)

        #Evaluate on the same chart for info (case (1))
        evaluator = ProfileEvaluator(
            chart_tif=args.build_tif,
            chart_type=args.build_type,
            chart_cie=args.build_cie,
            in_icc=creator.in_icc,
            # out_icc=args.out_icc,
            folder=args.out_folder,
            patch_data=creator.df,
            logger_name=CLI_LOGGER_NAME,
        )
        evaluator.evaluate_profile(fiducial_list_build, 
                                   report_title="Profile Creation Report", 
                                   report_filename="profile_creation_report.pdf"
        )

        #Evaluate profile on second chart if available (recommended; case (3))
        if (args.test_type != args.build_type):
            evaluator = ProfileEvaluator(
                chart_tif=args.test_tif,
                chart_type=args.test_type,
                chart_cie=args.test_cie,
                in_icc=creator.in_icc,
                # out_icc=args.out_icc,
                folder=args.out_folder,
                patch_data=None,
                logger_name=CLI_LOGGER_NAME,
            )
            evaluator.evaluate_profile(fiducial_list_test, 
                                       report_title="Profile Evaluation Report", 
                                       report_filename="profile_evaluation_report.pdf"
            )
            
    else: 
     #Evaluate only (no build_tif given; case (2))
        evaluator = ProfileEvaluator(
            chart_tif=args.test_tif,
            chart_type=args.test_type,
            chart_cie=args.test_cie,
            in_icc=args.in_icc,
            # out_icc=args.out_icc,
            folder=args.out_folder,
            patch_data=None,
            logger_name=CLI_LOGGER_NAME,
        )
        evaluator.evaluate_profile(fiducial_list_test,  report_filename="profile_evaluation_report.pdf")
