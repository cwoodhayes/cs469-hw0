"""
Code for loading in the experimental data
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import pathlib

import pandas as pd


@dataclass
class Dataset:
    """
    Container for the data in the UTIAS Multi-Robot Cooperative Localization and Mapping Datasets

    Originally sourced from here: http://asrl.utias.utoronto.ca/datasets/mrclam/index.html
    And included in the course canvas website here: https://canvas.northwestern.edu/courses/239182/files/folder/Datasets
    """

    barcodes: pd.DataFrame
    control: pd.DataFrame
    ground_truth: pd.DataFrame
    landmarks: pd.DataFrame
    measurement: pd.DataFrame

    # measurement file, but using the landmark ground truth
    # subject labels (translated using barcodes)
    # (this way landmarks & measurements can be used together more easily)
    measurement_fix: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        # rectify labels between measurement & ground truth

        # in barcodes data file:
        # "subject" = landmark subject  <- what we want to use (cuz they're nicer numbers)
        # "barcode" = measurement subject
        bc = self.barcodes

        msr_subj_to_landm_subj = pd.Series(
            bc["subject"].values, index=bc["barcode"].values
        )

        self.measurement_fix = self.measurement.copy()
        self.measurement_fix["subject"] = self.measurement["subject"].map(
            msr_subj_to_landm_subj
        )

    @classmethod
    def from_dataset_directory(cls, p: pathlib.Path) -> Dataset:
        """
        Load in

        :param p: path to the dataset directory
        :return: Dataset instance
        """
        # grab the prefix from directory name
        # this isn't particularly robust but it works fine for
        # this assignment
        prefix = p.stem
        barcodes = pd.read_csv(
            p / f"{prefix}_Barcodes.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["subject", "barcode"],
        )

        control = pd.read_csv(
            p / f"{prefix}_Control.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "forward_velocity_mps", "angular_velocity_radps"],
        )

        groundtruth = pd.read_csv(
            p / f"{prefix}_Groundtruth.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "x_m", "y_m", "orientation_rad"],
        )

        landmark = pd.read_csv(
            p / f"{prefix}_Landmark_Groundtruth.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["subject", "x_m", "y_m", "x_std_dev", "y_std_dev"],
        )

        measurement = pd.read_csv(
            p / f"{prefix}_Measurement.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "subject", "range_m", "bearing_rad"],
        )

        return cls(barcodes, control, groundtruth, landmark, measurement)
