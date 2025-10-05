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


class MeasurementModel:
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
            # set subject to be the same
            z.loc[idx, "subject"] = mark["subject"]

            # get vector pointing from robot to landmark
            p_landmark = np.array((mark["x_m"], mark["y_m"]))
            r_vec = p_landmark - x[0:2]
            r = np.linalg.norm(r_vec)

            # get unit vector of robot's POV
            robot_pov_vec = np.array((np.cos(x[2]), np.sin(x[2])))

            # get angle between robot POV and the robot->landmark vector
            cos_theta = robot_pov_vec.dot(r_vec) / r
            theta = np.arccos(cos_theta)
            # arccos always evaluates to theta <= pi radians; we need to
            # catch pi<theta<=2pi ourselves
            # can use the cross product here
            if np.cross(robot_pov_vec, r_vec) < 0:
                theta = -theta

            z.loc[idx, "range_m"] = r
            z.loc[idx, "bearing_rad"] = theta

        return z
