"""
Runner for particle filter using the test data
"""

import numpy as np
import pandas as pd
from hw0.data import Dataset, Trajectory
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import ParticleFilter


class ParticleFilterRunner:
    """
    Class to manage running a particle filter easily
    """

    def __init__(self):
        pass

    def run(self, ds: Dataset, pf: ParticleFilter, name: str) -> Trajectory:
        """
        Runs the given particle filter on the given dataset, returning
        a resulting trajectory

        :param ds: dataset
        :param pf: particle filter
        :param name: descriptive name for this run
        """
        # simulate the robot's motion
        # clump together measurements for each control.

        # note that we actually want the measurements _after_ each control (since they are
        # the ones taken while we are executing the command, and which are most relevant
        # to the state after that command)
        # so we insert a dummy control prior to the first one, at the same timestamp
        # as the first measurement
        states = []
        dummy_t0 = ds.measurement_fix["time_s"].iloc[0]
        if ds.control["time_s"].iloc[0] < dummy_t0:
            dummy_t0 = ds.control["time_s"].iloc[0]
        dummy_t0 -= 0.0001
        dummy_u0 = pd.DataFrame(
            {
                "time_s": [dummy_t0],
                "forward_velocity_mps": [0.0],
                "angular_velocity_radps": [0.0],
            }
        )
        control = pd.concat([dummy_u0, ds.control], ignore_index=True, copy=True)
        control["forward_velocity_mps"] /= 10

        print("Simulating...")
        for idx in range(0, len(control) - 1):
            ctl = control.iloc[idx]

            t_ = ctl["time_s"]
            t_next = control["time_s"].iloc[idx + 1]
            dt = t_next - t_

            not_late = ds.measurement_fix["time_s"] <= t_next
            not_early = ds.measurement_fix["time_s"] > t_
            meas = ds.measurement_fix[not_late & not_early]

            x_t = pf.step(control=ctl, measurements=meas, dt=dt)
            states.append(x_t)

        print("Done simulating. ")

        states = np.array(states)

        traj = pd.DataFrame(
            {
                "time_s": control["time_s"].iloc[1:],
                "x_m": states[:, 0],
                "y_m": states[:, 1],
                "orientation_rad": states[:, 2],
            }
        ).reset_index()

        return Trajectory(traj, name)


def dead_reckoner(ds: Dataset) -> Trajectory:
    """
    Docstring for dead_reckoner

    :param ds: dataset
    :return: trajectory for a dead-reckoned robot (i.e. raw dynamics from controls)
    :rtype: DataFrame in form of ds.ground_truth
    """
    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy()[0]

    # grab the commands
    u_ts = ds.control["time_s"].to_numpy()
    u = ds.control[["forward_velocity_mps", "angular_velocity_radps"]].to_numpy()
    u = u / (10, 1)

    states = []
    m = TextbookMotionModel()

    # simulate the robot's motion
    # must skip the last command cuz we don't know how long it runs
    x_prev = x_0
    for idx in range(u.shape[0] - 1):
        states.append(x_prev)
        dt = u_ts[idx + 1] - u_ts[idx]
        x_t = m.step(u[idx], x_prev, dt)
        x_prev = x_t
    states.append(x_prev)

    states = np.array(states)

    traj = pd.DataFrame(
        {
            "time_s": ds.control["time_s"],
            "x_m": states[:, 0],
            "y_m": states[::, 1],
            "orientation_rad": states[:, 2],
        }
    )
    return Trajectory(traj, "dead_reckoning")
