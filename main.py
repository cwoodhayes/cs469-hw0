"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib

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
    states = []

    for idx in range(commands.shape[0]):
        states.append(m.step(commands[idx], 1.0))

    states = np.array(states)

    # plot the resulting states over time
    plt.plot(states[:, 0], states[:, 1], marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q1: Robot path over 5 example commands")
    plt.grid(True)
    plt.show()

    # TODO - add quivers pointing out of points for heading
    # TODO - repeat this plot, but divide each command into 10 timesteps to show
    # the curves


if __name__ == "__main__":
    main()
