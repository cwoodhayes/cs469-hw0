"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

from hw0.data import Dataset
from hw0.motion import MotionModel

REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")

    # my assigned dataset is ds1, so I'm hardcoding this for now
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")
    question_1(ds)


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
    states = [m.INITIAL_STATE]

    for idx in range(commands.shape[0]):
        states.append(m.step(commands[idx], 1.0))

    states = np.array(states)
    ax = plt.subplot()
    plot_robot_path(states, 1.0, ax)
    plt.show()


def plot_robot_path(x_all: np.ndarray, dt: float | np.ndarray, ax: Axes) -> None:
    """
    Plots the path of the robot given a sequence of states & a time interval between them

    :param x_all: a sequence of t states ordered in time
    :type x_all: np.ndarray [x, y, theta]
    :param dt: time step between each state (or, an array of t-1 such timesteps)
    :type dt: float | np.ndarray [t_1 - t_0, t_2 - t_1, ... t_n - t_(n-1)]
    :param axes: axes object on which we should plot
    """
    # plot each robot location
    ax.plot(
        x_all[:, 0],
        x_all[:, 1],
        linestyle="-",
        marker="o",
        color="black",
    )

    # make arrows for each orientation
    headings = np.column_stack(
        (
            np.cos(x_all[:, 2]),
            np.sin(x_all[:, 2]),
        )
    )

    # mark each orientation on the locations with an arrow
    ax.quiver(
        x_all[:, 0],
        x_all[:, 1],  # arrow bases
        headings[:, 0],  # arrow directions
        headings[:, 1],
        angles="xy",
        scale_units="xy",
        color="red",
    )

    ax.set_title("Q1: Robot path over 5 example commands")
    ax.grid(True)
    # ax.set_aspect("equal")

    # TODO - repeat this plot, but divide each command into 10 timesteps to show
    # the curves


if __name__ == "__main__":
    main()
