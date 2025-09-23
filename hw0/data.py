"""
Code for loading in the experimental data
"""

from __future__ import annotations

from dataclasses import dataclass
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
    landmark_ground_truth: pd.DataFrame
    measurement: pd.DataFrame

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
            sep="\s+",
            engine="python",
            comment="#",
            names=["subject", "barcode"],
        )

        control = pd.read_csv(
            p / f"{prefix}_Control.dat",
            sep="\s+",
            engine="python",
            comment="#",
            names=["time_s", "forward_velocity_mps", "angular_velocity_radps"],
        )

        groundtruth = pd.read_csv(
            p / f"{prefix}_Groundtruth.dat",
            sep="\s+",
            engine="python",
            comment="#",
            names=["time_s", "x_m", "y_m", "orientation_rad"],
        )

        landmark = pd.read_csv(
            p / f"{prefix}_Landmark_Groundtruth.dat",
            sep="\s+",
            engine="python",
            comment="#",
            names=["subject", "x_m", "y_m", "x_std_dev", "y_std_dev"],
        )

        measurement = pd.read_csv(
            p / f"{prefix}_Measurement.dat",
            sep="\s+",
            engine="python",
            comment="#",
            names=["time_s", "subject", "range_m", "bearing_rad"],
        )

        return cls(barcodes, control, groundtruth, landmark, measurement)
