"""
Tests that are meant to be run standalone without pytest
"""

from matplotlib import pyplot as plt
import numpy as np
from hw0.data import Dataset
from hw0.motion import TextbookMotionModel
from hw0.plot import plot_robot_simple


def circle_test(ds: Dataset) -> None:
    # test for my own reassurance.
    # make the robot go in a big circle. should do it exactly once

    r = 10  # radius
    steps = 50  # number of steps to trace the circle
    dt = 1

    w = (2 * np.pi) / steps
    v = r * w * dt

    m = TextbookMotionModel()
    states = []

    commands = np.ones(shape=(steps, 2)) * (v, w)
    print(commands)

    x_prev = np.array([0.0, 0.0, 0.0])
    for idx in range(commands.shape[0]):
        states.append(x_prev)
        x_t = m.step(commands[idx], x_prev, 1.0)
        x_prev = x_t
    states.append(x_prev)

    states = np.array(states)
    ax = plt.subplot()
    plot_robot_simple(states, ax)
    ax.set_title("Circle Test")

    # some quick simple tests
    expected_angles = np.linspace(0, 2 * np.pi, steps)
    print("\nangles")
    print(states[:, 2].round(2))
    print("\n expected angles")
    print(expected_angles.round(2))

    # check that the lengths of all the segments are the same
    rotated = np.zeros_like(states)
    rotated[0, :] = states[-1, :]
    rotated[1:, :] = states[:-1, :]
    diffs = states[:, 0:2] - rotated[:, 0:2]
    distances = np.linalg.norm(diffs, axis=1)
    print("\ndistances (should b same)")
    print(np.round(distances, decimals=1))

    plt.show()
