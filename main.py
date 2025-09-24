"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from hw0.data import Dataset
from hw0.motion import MotionModel
from hw0.plot import plot_robot_path

REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")

    # my assigned dataset is ds1, so I'm hardcoding this for now
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")
    # question_1(ds)
    question_2(ds)


def question_1(ds: Dataset) -> None:
    m = MotionModel()

    commands = np.array(
        (
            (0.5, 0),
            (0, -1 / (2 * np.pi)),
            (0.5, 0),
            (0, 1 / (2 * np.pi)),
            (0.5, 0),
        )
    )
    states = [m.DEFAULT_INITIAL_STATE]

    for idx in range(commands.shape[0]):
        states.append(m.step(commands[idx], 1.0))

    states = np.array(states)
    ax = plt.subplot()
    plot_robot_path(states, 1.0, ax)
    ax.set_title("Q1: Robot path over 5 example commands")
    plt.show()


def question_2(ds: Dataset) -> None:
    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy()[0]
    # and the first timestamp from the controls
    t_0 = ds.control["time_s"][0]

    # grab the commands. they are already in an ok format for us
    u_ts = ds.control["time_s"].to_numpy()
    u = ds.control[["forward_velocity_mps", "angular_velocity_radps"]].to_numpy()

    states = [x_0]
    m = MotionModel(x_0, t_0)

    for idx in range(u.shape[0]):
        states.append(m.step_abs_t(u[idx], u_ts[idx]))

    states = np.array(states)
    ax_control = plt.subplot(1, 2, 1)
    plot_robot_path(
        states,
        u_ts,
        ax_control,
        show_orientations=False,
        show_points=False,
    )
    ax_control.set_title("robot trajectory predicted from control input")

    # make a plot for the actual ground truth values
    # the timestamps will be different but the trajectory should still be the same.
    ax_truth = plt.subplot(1, 2, 2)
    plot_robot_path(
        ds.ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy(),
        ds.ground_truth["time_s"].to_numpy(),
        ax_truth,
        show_orientations=True,
    )
    ax_truth.set_title("ground truth trajectory")

    plt.show()


if __name__ == "__main__":
    main()
