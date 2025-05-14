import argparse
import logging
import importlib.resources
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
PROFILES_BASE_PATH = importlib.resources.files("arte_profiler") / "data" / "profiles"

class BaseColorManager:
    def __init__(
        self,
        chart_tif: Union[str, Path],
        chart_type: str = "ColorCheckerSG",
        folder: Optional[Union[str, Path]] = None,
    ):
        self.chart_tif = Path(chart_tif)
        if folder is None:
            folder = "."
        self.folder = Path(folder)
        self.chart_type = chart_type
        self.logger, self.command_logger = profiling_utils.generate_logger(self.folder)
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
        self.chart_cie = TARGETS_BASE_PATH / self.reference_data["chart_cie"]

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
            self.logger.info(
                f"Setting up fiducial marks detection on {self.chart_tif}..."
            )

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
            str(self.folder / f"diag_{self.chart_type}.tiff"),
        ]

        self.logger.info(
            f"Running scanin to extract the RGB values of the patches from {self.chart_tif}..."
        )
        profiling_utils.run_command(
            scanin_cmd, self.command_logger
        )  # FIXME: add check the the command worked (try except ? see log)

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

    def detect_and_extract(self, fiducial: list = None):
        if fiducial == None:
            fiducial = list(self.find_fiducial().flatten())
        self.extract_rgb_values(fiducial=fiducial)


class ProfileCreator(BaseColorManager):
    """ """

    def __init__(self, chart_tif, chart_type, folder):
        super().__init__(chart_tif, chart_type, folder)

    def icc_from_ti3(self):
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
            "g",
            "-b",
            "n",
            "-q",
            "m",
            "-O",
            str(self.folder / "input_profile.icc"),
            str(self.chart_ti3.with_suffix("")),
        ]
        self.logger.info("Running colprof to build an input ICC profile...")
        profiling_utils.run_command(colprof_cmd, self.command_logger)
        self.in_icc = self.folder / "input_profile.icc"

    def build_profile(self, fiducial: list = None):
        self.logger.info(
            f"Profile generation through {self.chart_type} chart initialized."
        )
        self.detect_and_extract(fiducial=fiducial)
        self.icc_from_ti3()
        self.logger.info(f"Profile generated: {self.in_icc}")


class ProfileEvaluator(BaseColorManager):
    """ """

    def __init__(self, chart_tif, chart_type, in_icc, out_icc, folder, patch_data=None):
        super().__init__(chart_tif, chart_type, folder)
        self.in_icc = Path(in_icc)
        self.out_icc = out_icc
        self.df = patch_data

        with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
            profiles = yaml.safe_load(f)
        if out_icc in profiles.keys():
            self.out_icc = PROFILES_BASE_PATH / profiles[out_icc]["path"]
        else:
            self.out_icc = Path(out_icc)

        # Check if all files exist
        for file_path in [
            self.in_icc,
            self.out_icc,
        ]:
            if not file_path.is_file():
                raise FileNotFoundError(f"File {file_path} not found.")
            
    def get_corrected_lab_vals(self, use_pyvips: bool = False):
        """
        Compute corrected Lab values for the color chart patches using the
        input and output ICC profiles.

        Returns
        -------
        np.ndarray
            Array of corrected Lab values for the patches.
        """
        if use_pyvips:
            RGB = self.df[["RGB_R", "RGB_G", "RGB_B"]].values / 100
            RGB = (RGB * 65535).astype("uint16")

            corr_lab_vals = (
                pyvips.Image.new_from_array(RGB[None, ...])
                .icc_import(
                    input_profile=self.in_icc
                ) # ok: only one transform is set now (abs colorimetric). Careful when using LUT: It's marked as A2B0 but it's absolute colorimetric, so must tell pyvips perceptual.
                .numpy()
                .squeeze()
            )
                
            #     .icc_export(
            #         output_profile=str(self.out_icc), depth=16
            #     )  # ok: eciRGB v2 is a matrix-based working space with effectively one transform.
            #     .icc_import(pcs="lab", input_profile=self.out_icc)
            #     .numpy()
            #     .squeeze()  # ok: same as above for eciRGB v2.
            # )
        
        else:
            #write the DT-NGT2 RGB values in a .txt file (for input to icclu)
            self.df.to_csv(self.folder / "icclu_input_values.txt", sep='\t',
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

            self.logger.info(
                f"Running icclu..."
            )
            profiling_utils.run_command(
                icclu_cmd,
                self.command_logger,
                stdin_path= self.folder / "icclu_input_values.txt",
                stdout_path=self.folder / "icclu_output_values.txt",
            )  # FIXME: add check the the command worked (try except ? see log)

            corr_lab_vals = np.loadtxt(self.folder / "icclu_output_values.txt")

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
        self.logger.info(
            "Reading the corrected and ground truth values of the patches..."
        )
        gt_lab_vals = self.get_gt_lab_vals()
        corr_lab_vals = self.get_corrected_lab_vals(use_pyvips=False)

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

    def create_patch_comparison_chart(self):
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

        # sRGB = self.df[["RGB_R", "RGB_G", "RGB_B"]].to_numpy() / 100
        # sRGB = (sRGB * 255).astype("uint8")
        sRGB = colour.XYZ_to_sRGB(
            colour.Lab_to_XYZ(
                self.df[["gt_L", "gt_A", "gt_B"]].values,
                illuminant=np.array([0.3457, 0.3585]),
            ),
            illuminant=np.array([0.3457, 0.3585]),
        ).clip(0, 1)
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
                            self.reference_data["rows"], self.reference_data["cols"]
                        )[
                            row, col
                        ],  # FIXME: I swapped rows and cols, check
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
        # plt.subplots_adjust(left=0.5, right=0.5, top=0.5, bottom=0.5)
        plt.title(rf"$\Delta{{E}}_{{00}}^{{*}}$ for the patches", fontsize=16)
        fig1.tight_layout()
        fig1.savefig(self.folder / f"delta_e_{self.chart_type}.png", facecolor="w", dpi=dpi)
        plt.close(fig1)

    def create_delta_e_histogram(self):
        self.delta_e_hist_size = self.delta_e_size #(1000, 1000)
        dpi = 100
        fig = plt.figure(
            figsize=(self.delta_e_hist_size[0] / dpi, self.delta_e_hist_size[1] / dpi)
        )
        ax = fig.add_subplot(111)
        ax.hist(self.de_2000, bins=20, range=(0, 4))
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)
        ax.text(
            0.7,
            0.95,
            f"$\Delta{{E}}_{{00}}^{{*}}$ mean: {self.de_2000.mean():.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ 90%: {np.quantile(self.de_2000, 0.90):.2f} \n$\Delta{{E}}_{{00}}^{{*}}$ max: {self.de_2000.max():.2f}",
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
        fig.savefig(self.folder / f"stdev_patches_{self.chart_type}.png", facecolor="w", dpi=dpi)
        plt.close(fig)

    def generate_report(self, filename = "profiling_report.pdf"):  # FIXME: imgs shapes ok? based on 10x14?
        """
        Generate a PDF report summarizing the analysis results.

        The report includes:
        - ΔE color comparison charts.
        - A histogram of ΔE values.
        - Heatmaps of the standard deviation of RGB values.
        - Metadata and conclusions based on FADGI and Metamorfoze guidelines.
        """
        self.logger.info("Generating report...")
        c = canvas.Canvas(str(self.folder / filename), pagesize=A4)

        canvas_width, canvas_height = A4

        # title
        c.setFont("DejaVuSans-Bold", 12)
        c.drawString(100, 800, "Profiling Report")
        c.setFont("DejaVuSans", 11)
        t = datetime.fromtimestamp(time.time())
        c.drawString(100, 780, f"Generated on {t.date()} at {str(t.time())[:-7]}")
        c.drawString(100, 760, f"{self.chart_type} chart in image {self.chart_tif.name}")
        c.drawString(100, 740, f"Profile used: {self.in_icc.name}")
 
        # color accuracy
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 700, f"Color accuracy")

        c.drawImage(
            self.folder / f"delta_e_{self.chart_type}.png",
            100,
            400,
            width=self.delta_e_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )
        c.drawImage(
            self.folder / f"delta_e_hist_{self.chart_type}.png",
            100,
            100,
            width=self.delta_e_hist_size[0] // 3.5,
            height=self.delta_e_hist_size[1] // 3.5,
        )

        de_76_mean = self.de_76.mean()
        de_76_max = self.de_76.max()
        de_2000_mean = self.de_2000.mean()
        de_2000_quantile = np.quantile(self.de_2000, 0.90)

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 80, f"ΔE* mean: {de_76_mean:.2f}, ΔE* max: {de_76_max:.2f}")
        c.drawString(
            100,
            60,
            f"ΔE₀₀* mean: {de_2000_mean:.2f}, ΔE₀₀* 90%: {de_2000_quantile:.2f}",
        )

        if (de_76_mean <= 4.0) and (de_76_max <= 10.0):
            c.setFillColor("green")
            c.drawString(350, 80, "Metamorfoze")
        else:
            c.setFillColor("red")
            c.drawString(350, 80, "Metamorfoze")

        if (de_2000_mean <= 2.0) and (
            de_2000_quantile <= 4
        ):  # FADGI 2023: Paintings and Other Two-Dimensional Art (Other Than Prints) #FIXME
            c.setFillColor("green")
            c.drawString(350, 60, "FADGI")
        else:
            c.setFillColor("red")
            c.drawString(350, 60, "FADGI")

        c.showPage()

        # appendix
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(100, 800, f"Appendix")
        c.setFont("DejaVuSans", 11)
        c.drawString(
            100, 780, f"Standard deviation of the extracted RGB values of the patches"
        )
        c.drawImage(
            self.folder / f"stdev_patches_{self.chart_type}.png",
            100,
            100,
            width=self.stdev_patches_size[0] // 3,
            height=self.stdev_patches_size[1] // 3,
        )
        c.showPage()

        c.setFont("DejaVuSans", 11)
        c.drawString(100, 780, f"Extracted patches")
        diag = pyvips.Image.new_from_file(self.folder / f"diag_{self.chart_type}.tiff")
        diag = diag.thumbnail_image(1500)
        diag.write_to_file(self.folder / f"diag_{self.chart_type}.png")
        c.drawImage(
            self.folder / f"diag_{self.chart_type}.png",
            100,
            480,
            width=diag.width // 3.5,
            height=diag.height // 3.5,
        )

        # Save the PDF
        c.save()

    def make_plots(self):
        self.create_patch_comparison_chart()
        self.create_delta_e_histogram()
        self.plot_stdev_patches()

    def evaluate_profile(self, fiducial: list = None, report_filename: str = "profiling_report.pdf"):
        self.logger.info(
            f"Profile evaluation through {self.chart_type} chart initialized."
        )
        if self.df is None:
            self.detect_and_extract(fiducial)
        self.compute_delta_e()
        self.make_plots()
        self.generate_report(report_filename)
        self.logger.info(
            f"Evaluation report completed. Results saved in {self.folder}."
        )


def parse_args():
    """
    Parses the command-line arguments for arte-profiler, supporting:
      1) Build & Evaluate on the same chart
      2) Build on one chart, Evaluate on another chart
      3) Evaluate only (using a pre-existing ICC)
    """
    with open(TARGETS_BASE_PATH / "targets_manifest.yaml", "r") as f:
        available_targets = list(yaml.safe_load(f).keys())

    with open(PROFILES_BASE_PATH / "profiles_manifest.yaml", "r") as f:
        available_profiles = list(yaml.safe_load(f).keys())

    parser = argparse.ArgumentParser(
        prog="Profiling",
        description="""A Python wrapper around ArgyllCMS for camera profiling 
                    and color accuracy evaluation. The program supports (1) 
                    Building a color profile from an input image of a supported 
                    chart; (2) Evaluating a pre-existing icc profile through an 
                    input image of a supported chart; (3) Building and 
                    evaluating a color profile from an input image containg two 
                    different supported charts. The program produces a pdf 
                    report evaluating color accuracy against Metamorfoze and 
                    FADGI imaging guidelines.""",
    )

    parser.add_argument(
        "--build_tif",
        help="Path to the input image containing the target to build a color profile. Can contain one or two color charts. If omitted, no new profile is built.",
    )

    parser.add_argument(
        "--test_tif",
        help="Path to the input image containing the target to test a color profile. Can contain one or two color charts.",
        required=True,
    )

    parser.add_argument(
        "--build_type",
        choices=available_targets,
        help="Chart type to build an ICC profile from. If omitted, no new profile is built.",
    )

    parser.add_argument(
        "--test_type",
        choices=available_targets,
        help="Chart type to evaluate color accuracy on.",
        required=True,
    )

    parser.add_argument(
        "--fiducial_build",
        type=str,
        help="""Optional fiducial coords for the build target in the format x1,y1,x2,y2,x3,y3,x4,y4.
            This prevents auto-recognition of the chart and uses provided 
            marks instead.""",
    )

    parser.add_argument(
        "--fiducial_test",
        type=str,
        help="""Optional fiducial coords for the test target in the format x1,y1,x2,y2,x3,y3,x4,y4.""",
    )

    parser.add_argument(
        "--in_icc",
        type=str,
        help="A pre-existing ICC profile to evaluate. Must be provided when performing evaluation-only.",
    )

    parser.add_argument(
        "--out_icc",
        help=f"The output ICC profile. Can be one of: {available_profiles} or a specified path.",
        required=True,
    )

    # Output folder
    parser.add_argument(
        "-O",
        "--out_folder",
        type=str,
        default=".",
        help="The output folder where to write output from this program (default: current directory)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if not args.build_tif and not args.in_icc:
        parser.error(
            "You must provide an existing profile to test with --in_icc or build one with --build_tif and --build_type."
        )
    if (args.build_tif and not args.build_type) or ((args.build_type and not args.build_tif)):
        parser.error(
            "You must provide both --build_tif and --build_type to specify the build chart image path and type."
        )

    return args


def main():
    args = parse_args()

    # Convert potential fiducial coords into list of ints if provided
    fiducial_list_build = (
        list(map(int, args.fiducial_build.split(","))) if args.fiducial_build else None
    )
    fiducial_list_test = (
        list(map(int, args.fiducial_test.split(","))) if args.fiducial_test else None
    )

    # Basic logic:
    # 1) Build and test on same target if build_target is provided but test_target is None or test_target == build_target
    # 2) Build on one target and evaluate on a different one
    # 3) Evaluate only if build_target is None but in_icc is provided

    # Cases:
    if args.build_tif and args.build_type:
        creator = ProfileCreator(
            chart_tif=args.build_tif,
            chart_type=args.build_type,
            folder=args.out_folder,
        )
        creator.build_profile(fiducial_list_build)

        #Evaluate on the same chart for info
        evaluator = ProfileEvaluator(
            chart_tif=args.build_tif,
            chart_type=args.build_type,
            in_icc=creator.in_icc,
            out_icc=args.out_icc,
            folder=args.out_folder,
            patch_data=creator.df,
        )
        evaluator.evaluate_profile(fiducial_list_build, report_filename="profile_creation_report.pdf")

        if (args.test_type != args.build_type):
            #Evaluate on second chart (recommended)
            evaluator = ProfileEvaluator(
                chart_tif=args.test_tif,
                chart_type=args.test_type,
                in_icc=creator.in_icc,
                out_icc=args.out_icc,
                folder=args.out_folder,
                patch_data=None,
            )
            evaluator.evaluate_profile(fiducial_list_test, report_filename="profile_evaluation_report.pdf")
            

    # C) Evaluate only (no build_tif given)
    else: 
        evaluator = ProfileEvaluator(
            chart_tif=args.test_tif,
            chart_type=args.test_type,
            in_icc=args.in_icc,
            out_icc=args.out_icc,
            folder=args.out_folder,
            patch_data=None,
        )
        evaluator.evaluate_profile(fiducial_list_test,  report_filename="profile_evaluation_report.pdf")
        

    # else:
    #     # No build_target, no test_target => invalid usage,
    #     # or user didn't supply in_icc. We'll handle that.
    #     print("ERROR: Invalid combination of build/test arguments. "
    #           "Must provide at least one scenario: build_target or test_target (with in_icc).")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
