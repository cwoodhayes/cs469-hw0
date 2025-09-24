"""
Model for robot dynamics.
"""

import numpy as np


class MotionModel:
    """
    Represents the motion model for the robot
    """

    # x, y, \theta
    DEFAULT_INITIAL_STATE = (0.0, 0.0, 0.0)

    def __init__(self, x_0: np.ndarray = None, t_0: float = 0.0):
        if x_0 is None:
            x_0 = np.array(self.DEFAULT_INITIAL_STATE)
        self._state = x_0
        self.t_s = t_0

    def step_abs_t(self, control: np.ndarray, t: float) -> np.ndarray:
        """
        Same as step(), but a timestamp is given rather than a timestep.
        This must be > the previous timestamp.
        """
        dt = t - self.t_s
        return self.step(control, dt)

    def step(self, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Step forward dt in time

        :param control: Control input as a vector
        :type control: [forward velocity (m/s), angular velocity (rad/s)]
        :return: new state vector
        :rtype: [x, y, theta]
        """
        # quick and dirty implementation
        s = np.zeros_like(self._state)
        s_prev = self._state

        # radius r of the curved trajectory := |v|/w
        v = control[0]
        w = control[1]

        if w != 0:
            r = v / w

            s[0] = s_prev[0] + r * np.sin(w * dt) * np.tan(w * dt / 2)
            s[1] = s_prev[1] + r * np.sin(w * dt)
            s[2] = s_prev[2] + w * dt
        else:
            s[0] = s_prev[0] + (v * np.cos(s_prev[2]) * dt)
            s[1] = s_prev[1] + (v * np.sin(s_prev[2]) * dt)
            s[2] = s_prev[2]

        self._state = s
        self.t_s += dt
        return s
