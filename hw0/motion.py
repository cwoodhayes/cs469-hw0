"""
Model for robot dynamics.
"""

import numpy as np

from abc import ABC, abstractmethod


class NoiselessMotionModelBase(ABC):
    # x, y, \theta
    DEFAULT_INITIAL_STATE = (0.0, 0.0, 0.0)

    def __init__(self, x_0: np.ndarray = None, t_0: float = 0.0):
        if x_0 is None:
            x_0 = np.array(self.DEFAULT_INITIAL_STATE)
        self._state = x_0
        self.t_s = t_0

    @abstractmethod
    def step(self, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Step forward dt in time

        :param control: Control input as a vector
        :type control: [forward velocity (m/s), angular velocity (rad/s)]
        :return: new state vector
        :rtype: [x, y, theta]
        """
        pass

    def step_abs_t(self, control: np.ndarray, t: float) -> np.ndarray:
        """
        Same as step(), but a timestamp is given rather than a timestep.
        This must be > the previous timestamp.
        """
        dt = t - self.t_s
        return self.step(control, dt)


class NoiselessMotionModel(NoiselessMotionModelBase):
    """
    Represents the motion model for the robot
    """

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

        v = control[0]
        w = control[1]

        # theta_t = theta_t-1 + wdt
        s[2] = s_prev[2] + (w * dt)

        if (w * dt) == 0:
            # in this case, we have a straight line
            # (infinite radius, infinitesmal theta). handle that directly
            s[0] = s_prev[0] + (v * np.cos(s[2]))
            s[1] = s_prev[1] + (v * np.sin(s[2]))
            s[2] = s_prev[2]
        else:
            # radius r of the arc trajectory := |v|/wdt
            r = v / (w * dt)

            s[0] = s_prev[0] + r * (np.sin(s_prev[2]) - np.sin(s[2]))
            s[1] = s_prev[1] + r * (np.cos(s[2]) - np.cos(s_prev[2]))

        self._state = s
        self.t_s += dt
        return s


class TextbookNoiselessMotionModel(NoiselessMotionModelBase):
    """
    Motion model adapted from the Probabilistic robotics textbook
    Implemented for comparison against my own model above;

    Probabilistic Robotics, section 5.3, page 101
    """

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

        v = control[0]
        w = control[1]

        # theta_t = theta_t-1 + wdt
        s[2] = s_prev[2] + (w * dt)

        if (w * dt) == 0:
            # in this case, we have a straight line
            # (infinite radius, infinitesmal theta). handle that directly
            s[0] = s_prev[0] + (v * np.cos(s[2]))
            s[1] = s_prev[1] + (v * np.sin(s[2]))
            s[2] = s_prev[2]
        else:
            # radius r of the arc trajectory := |v|/wdt
            r = v / (w * dt)

            x_c = s_prev[0] - r * np.sin(s_prev[2])
            y_c = s_prev[1] + r * np.cos(s_prev[2])

            s[0] = x_c + r * (np.sin(s[2]))
            s[1] = y_c - r * (np.cos(s[2]))

        self._state = s
        self.t_s += dt
        return s
