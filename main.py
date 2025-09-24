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


if __name__ == "__main__":
    main()
