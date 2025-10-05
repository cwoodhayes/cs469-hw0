"""
Evaluation metrics for trajectories
generally for comparing prediction against ground truth
"""

import pandas as pd
import numpy as np


def interp2(
    traj1: pd.DataFrame, traj2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    simple linear interpolation, with each variable independent.
    warning: trajectories are modified in-place

    Since the groundtruth, control, and measurement data all have different
    timestamps, we in the error evals below must generally perform
    interpolation to be able to evaluate the trajectories point-to-point.
    Here we interpolate the higher-frequency trajectory to match the timestamps
    of the slower one before evaluating.

    :param traj: robot trajectory, in the same format as ds.groundtruth
    :param traj2: another robot trajectory
    :return: the same 2 trajectories, with the faster one interpolated so
    that its timestamps match the slower
    """
    if len(traj1) >= len(traj2):
        traj1_is_long = True
        long = traj1
        short = traj2
    elif len(traj1) < len(traj2):
        traj1_is_long = False
        long = traj2
        short = traj1

    new_long = short.copy()

    for col in ["x_m", "y_m", "orientation_rad"]:
        out = np.interp(
            x=short["time_s"],
            xp=long["time_s"],
            fp=long[col],
        )
        new_long[col] = out

    return (new_long, short) if traj1_is_long else (short, new_long)


def abs_trajectory_error(
    traj1: pd.DataFrame,
    traj2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate "Absolute Trajectory Error"
    aka the sum over all pointwise errors, divided by the number of samples

    Returns a dataframe of cumulative ATE at each timestamp on the lower-frequency
    trajectory

    timestamps for each traj must be the same
    """
    if not np.all(traj1["time_s"] == traj2["time_s"]):
        raise ValueError("Timestamps don't match.")

    x_t1 = traj1[["x_m", "y_m", "orientation_rad"]].to_numpy()
    x_t2 = traj2[["x_m", "y_m", "orientation_rad"]].to_numpy()

    err = np.linalg.norm(np.abs(x_t1 - x_t2), axis=1)
    ate = np.cumsum(err) / np.arange(1, x_t1.shape[0] + 1)

    # for each timestamp, sum over all prior timestamps

    return pd.DataFrame(
        {
            "time_s": traj1["time_s"],
            "ATE": ate,
        }
    )


def abs_trajectory_error_rmse(
    traj1: pd.DataFrame,
    traj2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Same as abs_trajectory_error except it uses RMSE rather than just magnitude
    """
    if not np.all(traj1["time_s"] == traj2["time_s"]):
        raise ValueError("Timestamps don't match.")

    x_t1 = traj1[["x_m", "y_m", "orientation_rad"]].to_numpy()
    x_t2 = traj2[["x_m", "y_m", "orientation_rad"]].to_numpy()

    err = np.linalg.norm(np.abs(x_t1 - x_t2), axis=1) ** 2
    ate = np.sqrt(np.cumsum(err) / np.arange(1, x_t1.shape[0] + 1))

    # for each timestamp, sum over all prior timestamps

    return pd.DataFrame(
        {
            "time_s": traj1["time_s"],
            "ATE_RMS": ate,
        }
    )
