import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
import yaml

import cv2
from . import profiling_utils
import shapely.geometry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
import seaborn as sns
import colour
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from typing import Union, Optional
import os


# Register fonts
pdfmetrics.registerFont(TTFont("DejaVuSans", str(Path(__file__).parents[2] / "tools" / "dejavu-sans_font" / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(Path(__file__).parents[2] / "tools" / "dejavu-sans_font" / "DejaVuSans-Bold.ttf")))

# Base path for reference data
TARGETS_BASE_PATH = Path(__file__).parents[2] / "data" / "targets"
PROFILES_BASE_PATH = Path(__file__).parents[2] / "data" / "profiles"

class ColorProfileBuilder:
    """
    A class for building color profiles from a color chart image and evaluating
    color accuracy.

    Attributes
    ----------
    chart_tif : Path
        Path to the input color chart image.
    chart_cht : Path
        Path to the chart recognition file.
    chart_cie : Path
        Path to the chart's ground truth Lab values file.
    out_icc : Path
        Path to the output ICC profile.
    folder : Path
        Directory for storing intermediate and output files.
    target_data : dict
        Dictionary containing target data.

    Methods
    -------
    __init__(chart_tif, chart_cht, chart_cie, out_icc, folder="."):
        Initialize the ColorProfileBuilder with paths to required files and the output folder.

    find_fiducial():
        Auto-recognize fiducial marks in the color chart.

    extract_rgb_values(fiducial=None):
        Extract RGB values from the color chart and return them as a DataFrame.

    build_icc_profile():
        Generate an ICC profile from the extracted RGB values.

    get_corrected_lab_vals():
        Compute corrected Lab values for the color chart patches.

    get_gt_lab_vals():
        Retrieve ground truth Lab values from the .cie file.

    compute_delta_e():
        Compute Delta E (CIE 1976 and CIE 2000) values for corrected vs. ground truth Lab values.

    plot_delta_e():
        Visualize Delta E values

    plot_stdev_patches():
        Generate heatmaps of the standard deviation of RGB values for the chart patches.

    generate_report():
        Generate a PDF report summarizing the analysis results, including Delta E and standard deviation heatmaps.

    run(fiducial=None):
        Perform the entire workflow: fiducial detection, RGB extraction, ICC profile generation,
        Delta E computation, plotting, and report generation.
    """

    def __init__(
        self,
        chart_tif: Union[str, Path],
        out_icc: Union[str, Path],
        chart_type: str = "ColorCheckerSG",
        folder: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the ColorProfileBuilder.

        Parameters
        ----------
        chart_tif : str or Path
            Path to the input color chart image.
        chart_type : str
            The color chart type. Default is "ColorCheckerSG".
        out_icc : str or Path
            Name of an available icc profile or path to an ICC profile.
        folder : str or Path, optional
            Directory to store intermediate and output files. If not provided, it defaults to ".".

        Raises
        ------
        FileNotFoundError
            If any of the required files (chart_tif, chart_cht, chart_cie, out_icc) are missing.
        """
        self.chart_tif = Path(chart_tif)
        if folder is None:
            folder = "."
        self.folder = Path(folder)
        self.chart_type = chart_type
        self.logger, self.command_logger = profiling_utils.generate_logger(self.folder)
        self.argyll_bin_path = profiling_utils.get_argyll_bin_path()

        with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
            profiles = yaml.safe_load(f)
        if out_icc in profiles.keys():
            self.out_icc = PROFILES_BASE_PATH / profiles[out_icc]["path"]
        else:
            self.out_icc = Path(out_icc)

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
        self.chart_cie = TARGETS_BASE_PATH / self.reference_data["chart_cie"]

        # Check if all files exist
        for file_path in [
            self.chart_tif,
            self.chart_cht,
            self.chart_cie,
            self.out_icc,
        ]:
            if not file_path.is_file():
                raise FileNotFoundError(f"File {file_path} not found.")

        # Create the directory and all parent directories if they don't exist
        self.folder.mkdir(parents=True, exist_ok=True)

    def find_fiducial(self, max_dim: int = 5000):
        """
        Auto-recognize fiducial marks in the color chart using SIFT.

        Parameters
        ----------
        max_dim : int, optional
            Maximum allowed image dimension for SIFT processing. If the image exceeds this,
            it will be downscaled to improve processing speed. Default is 5000.

        Returns
        -------
        np.ndarray
            Array of detected fiducial marks' coordinates.

        Raises
        ------
        RuntimeError
            If fiducial detection fails.
        """
        try:
            self.logger.info(f"Setting up fiducial marks detection on {self.chart_tif}...")

            sift = cv2.SIFT_create()
            reference = pyvips.Image.new_from_file(
                TARGETS_BASE_PATH / self.reference_data["image_path"]
            )[1].numpy()
            fiducial_ref = np.array(self.reference_data["fiducial"])
            kp1, ds1 = sift.detectAndCompute(reference, None)

            img2 = pyvips.Image.new_from_file(str(self.chart_tif))[1]
            # Check pixel dimensions and scale down if necessary
            scale_factor = 1
            if max(img2.width, img2.height) > max_dim:
                scale_factor = max_dim / max(img2.width, img2.height)
                self.logger.info(
                    f"Scaling image down by factor {scale_factor:.2f} for faster processing..."
                )
                img2 = img2.resize(scale_factor)

            self.logger.info(f"Determining the fiducial marks...")   
            img2 = ((img2.numpy() / 65535) * 255).astype("uint8")
            kp2, ds2 = sift.detectAndCompute(img2, None)

            if len(kp2) < 4:
                raise RuntimeError("Insufficient keypoints detected in target image.")

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(ds1, ds2, k=2)

            if not matches:
                raise RuntimeError("No matches found between reference and target image.")

            # store all the good matches as per Lowe's ratio test.
            good_m = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_m) < 4:
                raise RuntimeError("Not enough good matches to compute homography.")

            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_m]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_m]).reshape(
                -1, 1, 2
            )
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

            if M is None:
                raise RuntimeError("Failed to compute homography.")
            
            # Transform fiducial reference points
            fiducial_ref_h = np.concatenate(
                ((fiducial_ref), np.ones((4, 1))), axis=-1
            )
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
                raise RuntimeError("Detected fiducials are out of image bounds or invalid.")
            
        except Exception as e:
            self.logger.error(f"Fiducial detection failed: {e}")
            raise RuntimeError("Fiducial auto-recognition failed.") from e

    def extract_rgb_values(
        self,
        fiducial: list = None,
    ) -> pd.DataFrame:
        """
        Extract RGB values from the color chart image using ArgyllCMS' scanin.

        Parameters
        ----------
        fiducial : list, optional
            Coordinates of fiducial marks. If None, auto-detection will be attempted by scanin.

        Returns
        -------
        pd.DataFrame
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
            str(self.folder / "diag.tif"),
        ]

        self.logger.info(
            f"Running scanin to extract the RGB values of the patches from {self.chart_tif}..."
        )
        profiling_utils.run_command(scanin_cmd, self.command_logger)

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

    def build_icc_profile(self):
        """
        Generate an input ICC profile from the extracted RGB values using ArgyllCMS's
        colprof.

        Returns
        -------
        None

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
            "x",
            "-b",
            "n",
            "-q",
            "h",
            "-O",
            str(self.folder / "input_profile.icc"),
            str(self.chart_ti3.with_suffix("")),
        ]
        self.logger.info("Running colprof to build an input ICC profile...")
        profiling_utils.run_command(colprof_cmd, self.command_logger)
        self.in_icc = self.folder / "input_profile.icc"

    def get_corrected_lab_vals(self):
        """
        Compute corrected Lab values for the color chart patches using the
        input and output ICC profiles.

        Returns
        -------
        np.ndarray
            Array of corrected Lab values for the patches.
        """
        RGB = self.df[["RGB_R", "RGB_G", "RGB_B"]].values / 100
        RGB = (RGB * 65535).astype("uint16")

        corr_lab_vals = (
            pyvips.Image.new_from_array(RGB[None, ...])
            .icc_import(
                intent="perceptual", input_profile=self.in_icc
            )  # ok: only one device-to-PCS transform available. It's marked as A2B0 but it's absolute colorimetric.
            .icc_export(
                output_profile=str(self.out_icc), depth=16
            )  # ok: eciRGB v2 is a matrix-based working space with effectively one transform.
            .icc_import(pcs="lab", input_profile=self.out_icc)
            .numpy()
            .squeeze()  # ok: same as above for eciRGB v2.
        )

        self.df["corr_L"] = corr_lab_vals[:, 0]
        self.df["corr_A"] = corr_lab_vals[:, 1]
        self.df["corr_B"] = corr_lab_vals[:, 2]

        return corr_lab_vals

    def get_gt_lab_vals(self):
        """
        Retrieve ground truth Lab values for the color chart patches from the .cie file.

        Returns
        -------
        np.ndarray
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

    def compute_delta_e(self):
        """
        Compute Delta E values (CIE 1976 and CIE 2000) for corrected vs. ground
        truth Lab values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            DeltaE 1976 and DeltaE 2000 values for each patch.
        """
        self.logger.info("Reading the corrected and ground truth values of the patches...")
        gt_lab_vals = self.get_gt_lab_vals()
        corr_lab_vals = self.get_corrected_lab_vals()

        self.logger.info("Computing DeltaE...")
        self.de_76 = colour.difference.delta_E_CIE1976(gt_lab_vals, corr_lab_vals)
        self.de_2000 = colour.difference.delta_E_CIE2000(gt_lab_vals, corr_lab_vals)
        de_76_mean = self.de_76.mean()
        de_76_max = self.de_76.max()
        de_2000_mean = self.de_2000.mean()
        de_2000_quantile = np.quantile(self.de_2000, 0.90)

        self.logger.info(f"DE_76_mean: {de_76_mean:.2f}, DE_76_max: {de_76_max:.2f}")
        self.logger.info(
            f"DE_2000_mean: {de_2000_mean:.2f}, DE_2000_90th_percentile: {de_2000_quantile:.2f}"
        )
        if (de_76_mean > 4.0) or (de_76_max > 10.0):
            self.logger.warning(
                "Color accuracy is not compliant with Metamorfoze guidelines!"
            )
        if (de_2000_mean > 2.0) or (de_2000_quantile > 4):
            self.logger.warning(
                "Color accuracy is not compliant with FADGI guidelines!"
            )  # FADGI 2023: Paintings and Other Two-Dimensional Art (Other Than Prints)

        return self.de_76, self.de_2000

    def plot_delta_e(self):
        """
        Visualize Delta E values and create sRGB color charts comparing corrected and uncorrected colors.

        This method generates:
        1. A color chart comparing uncorrected and corrected colors for each patch, annotated with ΔE₀₀ values.
        2. A histogram of the ΔE₀₀ values.

        The charts are saved to the output folder.
        """
        # Visualize sRGB colors before correction (camera profile) and after correction (input profile just built)

        # convert corrected Lab values to sRGB (illuminant D50)
        corr_sRGB = colour.XYZ_to_sRGB(
            colour.Lab_to_XYZ(
                self.df[["corr_L", "corr_A", "corr_B"]].values,
                illuminant=np.array([0.3457, 0.3585]),
            ),
            illuminant=np.array([0.3457, 0.3585]),
        ).clip(0, 1)
        corr_sRGB = (corr_sRGB * 255).astype("uint8")

        sRGB = self.df[["RGB_R", "RGB_G", "RGB_B"]].to_numpy() / 100
        sRGB = (sRGB * 255).astype("uint8")

        # Visualize sRGB colors before correction (camera profile) and after correction (input profile just built)
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

                # Draw uncorrected rectangle
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
                )  # uncorrected

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
                )  # corrected

                # Calculate text position
                text_x = x1 + size / 2  # Center x
                text_y = (y1 + y2) / 2  # Center y

                text = str(
                    round(
                        self.de_2000.reshape(
                            self.reference_data["cols"], self.reference_data["rows"]
                        )[col, row],
                        2,
                    )
                )
                if len(text) < 4:
                    text = text + "0"

                # Draw text centered in rectangle
                ax1.text(
                    text_x,
                    text_y,
                    text,
                    fontsize=16,
                    color="white",
                    va="center",  # Center vertically
                    ha="center",  # Center horizontally
                    backgroundcolor=(0.4, 0.4, 0.4, 0.4),
                )
                index += 1

        ax1.imshow(img)
        ax1.axis("off")
        # plt.subplots_adjust(left=0.5, right=0.5, top=0.5, bottom=0.5)
        plt.title(f"$\Delta{{E}}_{{00}}^{{*}}$ for the patches", fontsize=16)
        fig1.tight_layout()
        fig1.savefig(self.folder / "delta_e.png", facecolor="w", dpi=dpi)
        plt.close(fig1)

        self.delta_e_hist_size = (1000, 1000)
        fig2 = plt.figure(
            figsize=(self.delta_e_hist_size[0] / dpi, self.delta_e_hist_size[1] / dpi)
        )
        ax2 = fig2.add_subplot(111)
        ax2.hist(self.de_2000, bins=20, range=(0, 4))
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)
        ax2.text(
            0.7,
            0.95,
            f"$\Delta{{E}}_{{00}}^{{*}}$ mean: {self.de_2000.mean():.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ 90%: {np.quantile(self.de_2000, 0.90):.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ max: {self.de_2000.max():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        ax2.set_xlabel(f"$\Delta{{E}}_{{00}}^{{*}}$", fontsize=12)
        ax2.set_ylabel("Number of patches", fontsize=12)

        fig2.tight_layout()
        fig2.savefig(self.folder / "delta_e_hist.png", facecolor="w", dpi=dpi)
        plt.close(fig2)

    def plot_stdev_patches(self):
        """
        Generate heatmaps of the standard deviation of RGB values for the chart patches.

        This method creates heatmaps for the R, G, and B standard deviation values of the extracted patches
        and saves them to the output folder.
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
        fig.savefig(self.folder / "stdev_patches.png", facecolor="w", dpi=dpi)
        plt.close(fig)

    def generate_report(self):  # FIXME: imgs shapes ok? based on 10x14?
        """
        Generate a PDF report summarizing the analysis results.

        The report includes:
        - ΔE color comparison charts.
        - A histogram of ΔE values.
        - Heatmaps of the standard deviation of RGB values.
        - Metadata and conclusions based on FADGI and Metamorfoze guidelines.
        """
        self.logger.info("Generating report...")
        c = canvas.Canvas(str(self.folder / "profiling_report.pdf"), pagesize=A4)

        canvas_width, canvas_height = A4

        # title
        c.setFont("DejaVuSans-Bold", 12)
        c.drawString(100, 800, f"Profiling Report for {self.chart_tif.name}")
        t = datetime.fromtimestamp(time.time())
        c.setFont("DejaVuSans", 11)
        c.drawString(100, 780, f"Generated on {t.date()} at {str(t.time())[:-7]}")

        # color accuracy
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 740, f"Color accuracy")

        c.drawImage(
            self.folder / "delta_e.png",
            100,
            440,
            width=self.delta_e_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )
        c.drawImage(
            self.folder / "delta_e_hist.png",
            100,
            140,
            width=self.delta_e_hist_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )

        de_76_mean = self.de_76.mean()
        de_76_max = self.de_76.max()
        de_2000_mean = self.de_2000.mean()
        de_2000_quantile = np.quantile(self.de_2000, 0.90)

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 120, f"ΔE* mean: {de_76_mean:.2f}, ΔE* max: {de_76_max:.2f}")
        c.drawString(
            100,
            100,
            f"ΔE₀₀* mean: {de_2000_mean:.2f}, ΔE₀₀* 90%: {de_2000_quantile:.2f}",
        )

        if (de_76_mean <= 4.0) and (de_76_max <= 10.0):
            c.setFillColor("green")
            c.drawString(350, 120, "Metamorfoze")
        else:
            c.setFillColor("red")
            c.drawString(350, 120, "Metamorfoze")

        if (de_2000_mean <= 2.0) and (
            de_2000_quantile <= 4
        ):  # FADGI 2023: Paintings and Other Two-Dimensional Art (Other Than Prints) #FIXME
            c.setFillColor("green")
            c.drawString(350, 100, "FADGI")
        else:
            c.setFillColor("red")
            c.drawString(350, 100, "FADGI")

        c.showPage()

        # appendix
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 800, f"Appendix")
        c.setFont("DejaVuSans", 11)
        c.drawString(
            100, 780, f"Standard deviation of the extracted RGB values of the patches"
        )
        c.drawImage(
            self.folder / "stdev_patches.png",
            100,
            100,
            width=self.stdev_patches_size[0] // 3,
            height=self.stdev_patches_size[1] // 3,
        )
        c.showPage()

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 780, f"Extracted patches")
        diag = pyvips.Image.new_from_file(self.folder / "diag.tif")
        diag = diag.thumbnail_image(1500)
        diag.write_to_file(self.folder / "diag.png")
        c.drawImage(
            self.folder / "diag.png",
            100,
            480,
            width=diag.width // 3.5,
            height=diag.height // 3.5,
        )

        # Save the PDF
        c.save()

    def run(self, fiducial: list = None):
        if fiducial == None:
            fiducial = list(self.find_fiducial().flatten())
        self.extract_rgb_values(fiducial=fiducial)
        self.build_icc_profile()
        # self.color_correct_chart()
        self.compute_delta_e()
        self.plot_delta_e()
        self.plot_stdev_patches()
        self.generate_report()
        self.logger.info(f"Profile and report creation completed. Results saved in {self.folder}.")


def parse_args():
    """
    Parses the command-line arguments for the Profiling tool.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguments.

    Command-line Arguments
    ----------------------
    --chart_tif: The color chart image (required)
    --out_icc: The output ICC profile (required)
    -F, --fiducial: List of fiducial marks to prevent auto-recognition of the chart (optional)
    -O, --out_folder : Path to the output folder where the results will be written (optional)
    """
    with open(TARGETS_BASE_PATH / "targets_manifest.yaml", "r") as f:
        available_targets = list(yaml.safe_load(f).keys())

    with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
        available_profiles = list(yaml.safe_load(f).keys())

    parser = argparse.ArgumentParser(
        prog="Profiling",
        description="""A Python wrapper around ArgyllCMS that builds an ICC 
                    color profile from an image of a Colorchecker Digital SG
                    card and produces a report evaluating color accuracy 
                    against Metamorfoze and FADGI imaging guidelines.""",
    )
    parser.add_argument("--chart_tif", help="The color chart image", required=True)
    parser.add_argument(
        "--chart_type", help="The chart type", choices=available_targets, required=True
    )
    parser.add_argument(
        "--out_icc",
        help=f"The output ICC profile. Can be one of: {available_profiles} or a specified path.",
        required=True,
    )
    parser.add_argument(
        "-F",
        "--fiducial",
        type=str,
        help="""Optional list of fiducial marks as x1,y1,x2,y2,x3,y3,x4,y4. 
             This prevents auto-recognition of the chart and uses provided 
             marks instead.""",
    )
    parser.add_argument(
        "-O",
        "--out_folder",
        type=str,
        help="The output folder where to write output from this program",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    fiducial_list = list(map(int, args.fiducial.split(","))) if args.fiducial else None

    builder = ColorProfileBuilder(
        chart_tif=args.chart_tif,
        chart_type=args.chart_type,
        out_icc=args.out_icc,
        folder=args.out_folder,
    )

    builder.run(fiducial_list)


if __name__ == "__main__":
    main()