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

    def __init__(self):
        self._alpha = 1.0
        # weighted by y = e^(-\alpha * dt)
        # where dt is time-since-measurement

    def z_given_x(self, x: np.ndarray) -> ZType:
        """
        Returns measurements z given current state x

        :param x: current state
        :type x: np.ndarray [x, y, theta]
        :return: z {subject #: (range (m), bearing (rads))}
        :rtype: ZType
        """

        out = pd.DataFrame(
            {
                "subject": [8],
                "x_m": [12.0],
                "y_m": [11.0],
            }
        )
        return out
