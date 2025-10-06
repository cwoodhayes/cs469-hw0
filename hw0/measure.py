"""
Measurement estimation
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm


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

    def __init__(self, landmarks: pd.DataFrame, cov_matrix: np.ndarray):
        """
        :param landmarks: ds.landmarks
        :param cov_matrix: gaussian measurement noise covariance matrix
        :type cov_matrix: 2x2 np.ndarray
        """
        self._landmarks = landmarks
        self._cov = cov_matrix

        # a more efficient representation for fast accesses
        lmdict = dict()
        for lm in self._landmarks.itertuples():
            lmdict[lm.subject] = np.array([lm.x_m, lm.y_m])
        self._lmdict = lmdict

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

    def z_given_x_by_landmark(self, x: np.ndarray, subject: int) -> np.ndarray:
        """
        More efficient function that just gets the measurement for a single landmark

        :param x: current state
        :type x: np.ndarray [x, y, theta]
        :param subject: subject number, per ds.measurements_fix
        :return: observation z for subject
        :rtype: ndarray [range_m, bearing_rad]
        """
        # get vector pointing from robot to landmark
        p_landmark = self._lmdict[subject]
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

        return np.array([r, theta])

    def probability_z_given_x(self, z_actual: tuple, x: np.ndarray) -> float:
        """
        Evaluates probability of observing z_actual, given state x

        Uses Gaussian likelihood

        :param z_actual: actual observation z
        :type z_actual: namedtuple of (time_s, subject, range_m, bearing_rad)
        :param x: [x_m, y_m, orientation_rad]
        :return: probability 0-1
        """
        z_pred = self.z_given_x_by_landmark(x, z_actual.subject)

        z = np.array((z_actual.range_m, z_actual.bearing_rad))
        err = z - z_pred
        err_T = err.reshape((-1, 1))

        exp = 0.5 * (err @ self._cov @ err_T)
        errs = expm(exp)

        # flatten out into a float for weighting
        norm_err = np.linalg.norm(errs)

        return norm_err
