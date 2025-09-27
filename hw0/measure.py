"""
Measurement estimation
"""

import numpy as np
import pandas as pd


ZType = pd.DataFrame
"""
each z is a list of locations of all landmarks,
in the same format as landmarks_groundtruth minus stddevs
("subject", "x_m", "y_m")
"""


class MeasurePredictor:
    """
    Predicts z given a noiseless x
    """

    def __init__(self, landmarks: pd.DataFrame):
        self._landmarks = landmarks

    def z_given_x(self, x: np.ndarray) -> ZType:
        """
        Returns measurements z given current state x

        :param x: current state
        :type x: np.ndarray [x, y, theta]
        :return: z {subject #: (range (m), bearing (rads))}
        :rtype: ZType
        """

        z = pd.DataFrame(
            np.nan,
            index=range(self._landmarks.shape[0]),
            columns=["subject", "range_m", "bearing_rad"],
        )

        for idx, mark in self._landmarks.iterrows():
            z["subject"][idx] = mark["subject"]
            landmark_loc = np.array(mark["x_m"], mark["y_m"])
            x_to_l = landmark_loc - x[0:2]
            r = np.linalg.norm(x_to_l)
            alpha = np.arcsin(x_to_l[1] / r)

            z["range_m"][idx] = r
            z["bearing_rad"][idx] = x[2] + alpha

        return z
