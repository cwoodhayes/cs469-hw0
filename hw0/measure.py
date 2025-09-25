"""
Measurement estimation
"""

import numpy as np


ZVec: np.ndarray
"""
each z is a list of all known landmark locations,
as well as the timestamp at which they were last seen by the robot.
"""


class ExpWeightedAverageMeasure:
    """
    Calculates p(z|x) by taking the weighted average of our state as implied
    by all measured landmark locations (individually), exponentially weighted by
    how long it's been since that location was last measured.
    """

    def __init__(self):
        self._alpha = 1.0
        # weighted by y = e^(-\alpha * dt)
        # where dt is time-since-measurement
