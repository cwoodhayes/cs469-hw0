"""
Code for the actual particle filter
"""

import abc
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from hw0.motion import NoiselessMotionModelBase
from hw0.measure import MeasurementModel


class GaussianProposalSampler:
    """
    Samples from gaussian centered at the origin
    """

    def __init__(self, shape: tuple[int], stddev: np.ndarray):
        """
        :param shape: output shape
        :type shape: tuple[int]
        """
        self.shape = shape
        self.stddev = stddev
        self.rng = np.random.default_rng()

    def sample(self) -> np.ndarray:
        return self.rng.normal(0, self.stddev, size=self.shape)


class ParticleFilter:
    @dataclass
    class Config:
        n_particles: int = 1000
        proposal_distribution: Literal["gaussian", "uniform"] = "gaussian"
        output_selection: Literal["weighted_mean"] = "weighted_mean"

    DEFAULT_CONFIG = Config()

    def __init__(
        self,
        motion_model: NoiselessMotionModelBase,
        measurement_model: MeasurementModel,
        X_0: np.ndarray | None,
        proposal_sampler: GaussianProposalSampler | None = None,
        config: Config = DEFAULT_CONFIG,
    ):
        """
        :param motion_model: dynamics model for the robot
        :param measurement_model: measurement model for the robot
        :param X_0: initial particle set, or a single known initial state
        :param proposal_sampler: noise generator
        :param config: filter configuration object
        """
        self.motion = motion_model
        self.measure = measurement_model
        self._c = config

        ## input checking
        if not (X_0.shape == (config.n_particles, 3) or X_0.shape == (3,)):
            raise ValueError(
                f"X_0 shape={X_0.shape} must be (3,) or {(config.n_particles, 3)}"
            )
        if config.n_particles <= 0:
            raise ValueError("n_particles must be >0")

        self._t = 0
        self._X_t = np.ndarray(shape=(config.n_particles, 3))
        self._X_t[:, :] = X_0
        # """each particle is (x_m, y_m, orientation_rads)"""

        # we'll be sampling noise for every particle
        if proposal_sampler is None:
            # TODO enable passing parameters in config
            self._noise = GaussianProposalSampler(
                shape=(config.n_particles, 3),
                stddev=0.4,
            )
        else:
            self._noise = proposal_sampler

    def step(self, control: pd.Series, measurements: pd.DataFrame) -> np.ndarray:
        """
        Process the filter, taking us from X_t-1 to X_t

        :param control: a single control input
        :type control: pd.Series, in the format of Dataset.control
        :param measurements: a set of measurement inputs taken in (t-1, t] -- i.e.
        the previous call of step() and the timestamp of control
        :type measurements: pd.DataFrame, in the format of Dataset.measurement
        :return: a single state prediction at time t (x_t)
        :rtype: ndarray[time_s, x_m, y_m, orientation_rad]
        """
        ## Input sanity checking
        new_t = control["time_s"]
        if new_t <= self._t:
            raise ValueError("t must increase monotonically")

        if len(measurements) == 0:
            # TODO figure out what to do here.
            # for now just skipping these steps entirely.
            print(f"No measurements for time ({self._t}, {new_t}]")
            return self._X_t[0]

        m_invalid = (measurements["time_s"] <= self._t) | (
            measurements["time_s"] > new_t
        )
        if any(m_invalid):
            print("Invalid measurement: ")
            print(measurements)
            raise ValueError("measurement timestamps are invalid")

        # TODO replace with the real filter:
        # just doing dead reckoning for now
        # with particles gaussian-distributed about the dead reckon result
        x = self.motion.step(control.to_numpy()[1:], new_t - self._t)
        self._X_t[:, :] = x
        self._X_t += self._noise.sample()

        ## Final updates
        self._t = new_t

        ## Derive concrete state prediction from particles
        # TODO
        return self._X_t[0]

    def get_Xt(self) -> np.ndarray:
        """
        Return the current particle set X_t, as determined by the most recent call of step()

        :return: particle set
        :rtype: [(x_m, y_m, orientation_rad), (...), ...]
        """
        return self._X_t

    def get_t(self) -> float:
        """
        Get time t at the most recent call of step()
        """
        return self._t
