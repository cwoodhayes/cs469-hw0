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
    """
    Particle filter implementation for turtle navigation (aka montecarlo localization)

    Implementation references:
    - Ch4 of Probabilistic Robotics (p78 for pseudocode)
    """

    @dataclass
    class Config:
        n_particles: int = 1000
        proposal_distribution: Literal["gaussian", "uniform"] = "gaussian"
        output_selection: Literal["weighted_mean"] = "weighted_mean"
        random_seed: int | None = None

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
                stddev=0.001,
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

        m_invalid = (measurements["time_s"] <= self._t) | (
            measurements["time_s"] > new_t
        )
        if any(m_invalid):
            print("Invalid measurement: ")
            print(measurements)
            raise ValueError("measurement timestamps are invalid")

        ## FILTER implementation
        # TODO be more clever about numpy usage?

        X_prev = self._X_t
        Xbar_t = np.empty(shape=(self._c.n_particles, 3))
        W_t = np.empty(shape=(self._c.n_particles,))  # particle weights for Xbar_t
        u_t = control.to_numpy()[1:]
        X_noise_t = self._noise.sample()

        ## CONTROL + MEASUREMENT UPDATE STEP
        for idx in range(self._c.n_particles):
            ## sample from p(x_t | u_t, x_t-1)

            # get dynamics prediction x_t(u_t, x_t-1)
            # i.e. propagate this one particle according to the deterministic
            # physics model
            x_t = self.motion.step_abs_t(u_t, new_t, x_prev=X_prev[idx])
            # add noise to get p(x_t | ...) AKA this particle
            x_t += X_noise_t[idx, :]

            ## weight the particle by p(z_t | x_t)

            # since we can have multiple measurements for one control, we
            # do this in a loop, such that we sum our probilities together.
            # TODO
            w_t = 1.0

            ## add this particle+weight (x_t, w_t) to Xbar_t
            Xbar_t[idx] = x_t
            W_t[idx] = w_t

        ## RESAMPLING STEP
        # normalize the weights so they sum to 1
        W_t /= np.sum(W_t)
        # resample using the weights
        X_t = self._sample_low_variance(Xbar_t, W_t)

        ## Final updates
        self._t = new_t
        self._X_t = X_t

        ## Derive concrete state prediction from particles
        match self._c.output_selection:
            case "weighted_mean":
                x = self._select_weighted_mean(X_t)
            case _:
                raise ValueError(
                    f"Unknown output selection config {self._c.output_selection}"
                )

        return x

    @staticmethod
    def _select_weighted_mean(X_t: np.ndarray) -> np.ndarray:
        """
        Select a representative value from the belief distribution
        represented by particle set X_t, using expected value

        Since this implementation represents particle density with frequency,
        not additional weights, this is just a straight-up mean.

        :param X_t: input particle set. each row is a state vector
        :return: x_t. Concrete state estimate at time t
        """
        return np.mean(X_t, axis=0)

    def _sample_low_variance(self, X_t: np.ndarray, W_t: np.ndarray) -> np.ndarray:
        """
        Low-variance sampling orientation per pseudocode in P.R. Ch 4 pg 86

        :param X_t: input particle set. each row is a state vector
        :param W_t: input weights for each particle. each row is a scalar weight
        :return: Xbar_t
        :rtype: X_t resampled using weights W_t
        """
        Xbar_t = []
        M = X_t.shape[0]

        r = self._noise.rng.uniform(0, 1 / M)
        c = W_t[0]
        i_ = 0
        for m in range(M):
            u = r + (m / M)
            while u > c:
                i_ += 1
                c += W_t[i_]
            Xbar_t.append(X_t[i_])

        return np.array(Xbar_t)

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
