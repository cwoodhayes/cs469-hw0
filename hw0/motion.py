"""
Model for robot dynamics.
"""

import numpy as np
from scipy.linalg import expm

from abc import ABC, abstractmethod


class NoiselessMotionModelBase(ABC):
    # x, y, \theta
    DEFAULT_INITIAL_STATE = (0.0, 0.0, 0.0)

    def __init__(self, x_0: np.ndarray = None, t_0: float = 0.0):
        if x_0 is None:
            x_0 = np.array(self.DEFAULT_INITIAL_STATE)
        self._state = x_0
        """x, y, theta"""

        self.t_s = t_0
        """initial timestamp"""

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


class TwistNoiselessMotionModel(NoiselessMotionModelBase):
    """
    Defines this motion as twists in SE(3) (with z=0), moves them into se(3), and operates
    on them using linear algebra.
    Implementing this now for funsies since i'm just learning this in ME449

    for quick math reference - see p99 in Modern Robotics by K. Lynch
    """

    @staticmethod
    def skew_symmetrify(v: np.ndarray) -> np.ndarray:
        """
        Get the skew symmetric matrix form [v] of v, such that
        v x a = [v]a

        :param v: a vector in R3
        """
        out = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )
        return out

    def _get_Ad_Tsb(self) -> np.ndarray:
        """
        get Ad_Tsb, the 6x6 adjoint matrix representation of T_sb, which is
        the transformation matrix that takes us from {s} to {b}

        :return: 6x6 matrix
        """
        # rotation matrix needed to rotate us into body frame
        # except writing it transposed cuz i find that easier to write down
        theta = self._state[2]
        R = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        ).T
        # translation component of T
        p_T = np.array([*self._state[:2], 0])
        p_ss = self.skew_symmetrify(p_T)

        # adjoint matrix of T
        Ad_T = np.zeros(shape=(6, 6))
        Ad_T[:3, :3] = R
        Ad_T[3:, :3] = p_ss @ R
        Ad_T[3:, 3:] = R

        return Ad_T

    def exp_twist(twist: np.ndarray, t: float) -> np.ndarray:
        """
        Given a 4x4 twist in se(3) and a duration t, apply the twist for t
        and return the resulting transformation as a 4x4 transformation
        matrix in SE(3)

        :param twist: se(3) twist (4x4, in bracket form)
        :type twist: np.ndarray
        :param t: duration for which the twist is applied
        :return: 4x4 SE(3) transformation matrix
        """
        # oh boy. i do not feel like writing this closed form out.
        # let's just use scipy's general matrix exp approximation method
        return expm(twist * t)

    def step(self, control: np.ndarray, dt: float):
        """
        Step forward dt in time

        :param control: Control input as a vector
        :type control: [forward velocity (m/s), angular velocity (rad/s)]
        :return: new state vector
        :rtype: [x, y, theta]
        """
        ## Get the twist that represents our current motion

        # define our twist V_b in body frame {b}
        # in this frame, x axis points forward, y axis points left, and z axis points up
        # the below is [omega_b, v_b]
        twist_b = np.array([0, 0, control[1], control[0], 0, 0])

        # transform it into space frame {s}
        twist_s = self._get_Ad_Tsb() * twist_b.reshape(-1, 1)

        # get matrix representation of this twist in {s} [V_s]
        omega_s = twist_s[:3]
        v_s = twist_s[3:]
        twist_bracket_s = np.zeros(shape=(4, 4))
        twist_bracket_s[:3, :3] = self.skew_symmetrify(omega_s)
        # TODO see if this works without reshape? so i understand broadcasting better
        twist_bracket_s[:3, 3] = v_s.reshape(-1, 1)

        ## Given the twist representing our motion, exponentiate it by t,
        ## which gives us our final transformation matrix T from time 0 to time t,
        ## here denoted T_0f
        T_0f = self.exp_twist(twist_bracket_s, dt)

        # apply this transform to our current location
        prev_loc = np.array([*self._state, 0]).reshape(-1, 1)
        new_loc = (T_0f @ prev_loc).T

        self._state = np.array([new_loc[0], new_loc[1], control[2] * dt])

        self.t_s += dt
        return self._state
