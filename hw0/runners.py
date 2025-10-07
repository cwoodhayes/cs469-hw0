"""
Runner for particle filter using the test data
"""

import pathlib
import subprocess
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import trange

from hw0.data import Dataset, Trajectory
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import ParticleFilter

__REPO_ROOT = pathlib.Path(__file__).parent.parent
DEFAULT_RUN_DIR = __REPO_ROOT / "data/filter-runs"
DR_TRAJECTORY_FILE = DEFAULT_RUN_DIR / "dr_traj.csv"

COMMIT_HASH = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class ParticleFilterRunner:
    """
    Class to manage running a particle filter easily
    """

    def __init__(self, run_dir: pathlib.Path = DEFAULT_RUN_DIR):
        self.run_dir = run_dir

    def run(
        self,
        ds: Dataset,
        pf: ParticleFilter,
        name: str = "run",
        write: bool = True,
        write_dr: bool = True,
    ) -> Trajectory:
        """
        Runs the given particle filter on the given dataset, returning
        a resulting trajectory.
        Writes the result to a directory

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
        t_start = time.monotonic()
        for idx in trange(0, len(control) - 1):
            ctl = control.iloc[idx]

            t_ = ctl["time_s"]
            t_next = control["time_s"].iloc[idx + 1]
            dt = t_next - t_

            not_late = ds.measurement_fix["time_s"] <= t_next
            not_early = ds.measurement_fix["time_s"] > t_
            meas = ds.measurement_fix[not_late & not_early]

            x_t = pf.step(control=ctl, measurements=meas, dt=dt)
            states.append(x_t)

        t_end = time.monotonic()
        run_duration = t_end - t_start
        print(f"Done simulating after {run_duration} secs.")

        states = np.array(states)

        df = pd.DataFrame(
            {
                "time_s": control["time_s"].iloc[1:],
                "x_m": states[:, 0],
                "y_m": states[:, 1],
                "orientation_rad": states[:, 2],
            }
        ).reset_index()

        traj = Trajectory(df, name)

        if write:
            dir_p = self._write(ds, pf, traj, name, run_duration)

            if write_dr:
                dead_reckoner(ds, write_path=dir_p)

        return traj

    def _write(
        self,
        ds: Dataset,
        pf: ParticleFilter,
        traj: Trajectory,
        name: str,
        run_duration: float,
    ) -> pathlib.Path:
        """
        Writes this trajectory & associated run info to a directory
        """
        dir_p = unique_path(self.run_dir / name)
        dir_p.mkdir(parents=True, exist_ok=False)

        print(f"Writing run info to {dir_p}...")

        traj.to_csv(dir_p, name)

        # pickle our pf
        with open(dir_p / f"{name}_pf.pkl", "wb") as f:
            pickle.dump(pf, f)

        # write a quick txt file describing the run
        txt = "Measurement Covariance: \n"
        txt += np.array2string(pf.measure._cov, precision=3, suppress_small=True)
        txt += "\n\n Control Noise: \n"
        txt += str(pf._u_noise.stddev)
        txt += "\n\n Git commit hash:\n"
        txt += COMMIT_HASH
        txt += f"\n\n Run duration: {run_duration} secs"

        with open(dir_p / f"{name}_desc.txt", "w") as f:
            f.write(txt)

        return dir_p

    def load(self, name: str) -> tuple[Trajectory, ParticleFilter, Trajectory, str]:
        """
        Loads in a prior run. returns traj, pf, name
        """
        dir_p = self.run_dir / name
        traj = Trajectory.from_file(dir_p / f"{name}_traj.csv")
        dr = Trajectory.from_file(dir_p / "dr_traj.csv")
        with open(dir_p / f"{name}_pf.pkl", "rb") as f:
            pf = pickle.load(f)

        return (traj, pf, dr, name)


def dead_reckoner(ds: Dataset, write_path: pathlib.Path = None) -> Trajectory:
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

    df = pd.DataFrame(
        {
            "time_s": ds.control["time_s"],
            "x_m": states[:, 0],
            "y_m": states[::, 1],
            "orientation_rad": states[:, 2],
        }
    )
    traj = Trajectory(df, "dead_reckoning")

    if write_path is not None:
        traj.to_csv(write_path, "dr")

    return traj


def unique_path(p: pathlib.Path) -> pathlib.Path:
    """
    return a path guaranteed not to overwrite an existing file.
    if p exists, append _2, _3, etc before the suffix.
    """
    if not p.exists():
        return p

    i = 2
    while True:
        candidate = p.parent / f"{p.stem}_{i}{p.suffix}"
        if not candidate.exists():
            return candidate
        i += 1
