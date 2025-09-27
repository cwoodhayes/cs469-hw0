"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hw0.data import Dataset
from hw0.motion import TextbookNoiselessMotionModel
from hw0.plot import plot_robot_path, plot_z_and_landmarks, plot_z_polar
from hw0.measure import MeasurePredictor

REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")
    # circle_test(ds)
    # question_2(ds)
    question_3(ds)
    question_6(ds)


def circle_test(ds: Dataset) -> None:
    # test for my own reassurance.
    # make the robot go in a big circle. should do it exactly once

    r = 10  # radius
    steps = 50  # number of steps to trace the circle
    dt = 1

    w = (2 * np.pi) / steps
    v = r * w * dt

    m = TextbookNoiselessMotionModel()
    states = [m.DEFAULT_INITIAL_STATE]

    commands = np.ones(shape=(steps, 2)) * (v, w)
    print(commands)

    for idx in range(commands.shape[0]):
        states.append(m.step(commands[idx], dt))

    states = np.array(states)
    ax = plt.subplot()
    plot_robot_path(states, dt, ax)
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


def question_2(ds: Dataset) -> None:
    print("!!!!!!!!!!!!!!!!!!! QUESTION 2 !!!!!!!!!!!")
    m = TextbookNoiselessMotionModel()

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
    fig, ax = plt.subplots(1, 1)
    plot_robot_path(states, 1.0, ax)
    ax.set_title("Q1: Robot path over 5 example commands")
    fig.canvas.manager.set_window_title("question_2")

    plt.show()


def question_3(ds: Dataset) -> None:
    print("!!!!!!!!!!!!!!!!!!! QUESTION 3 !!!!!!!!!!!")

    # this is for debugging purposes, to grab only a subset of the points
    # END_TIME = 1288971999.929
    END_TIME = 2288973229.039  # way past the end; uncomment to use all points
    ground_truth = ds.ground_truth[ds.ground_truth["time_s"] < END_TIME]
    control = ds.control[ds.control["time_s"] < END_TIME]
    # end debug stuff

    # grab the initial location from the first ground truth value
    x_0 = ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy()[0]
    # and the first timestamp from the controls
    t_0 = control["time_s"][0]

    # grab the commands
    u_ts = control["time_s"].to_numpy()
    u = control[["forward_velocity_mps", "angular_velocity_radps"]].to_numpy()
    u = u / (10, 1)

    states = [x_0]
    m = TextbookNoiselessMotionModel(x_0, t_0)

    # simulate the robot's motion
    for idx in range(u.shape[0]):
        states.append(m.step_abs_t(u[idx], u_ts[idx]))

    # plot the result
    states = np.array(states)
    fig, axes = plt.subplots(1, 2)
    ax_control = axes[0]
    plot_robot_path(
        states,
        u_ts,
        ax_control,
        show_orientations=False,
        show_points=False,
    )
    ax_control.set_title("Predicted Path from u")

    # make a plot for the actual ground truth values
    # the timestamps will be different but the trajectory should still be the same.
    ax_truth = axes[1]
    plot_robot_path(
        ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy(),
        ground_truth["time_s"].to_numpy(),
        ax_truth,
        show_orientations=True,
    )
    ax_truth.set_title("ground truth trajectory")
    fig.canvas.manager.set_window_title("question_3")

    plt.show()


def question_6(ds: Dataset, plot: bool = False) -> None:
    print("!!!!!!!!!!!!!!!!!!! QUESTION 6 !!!!!!!!!!!")
    test_data = pd.DataFrame(
        {
            "position": [
                (2, 3, 0),
                (0, 3, 0),
                (1, -2, 0),
            ],
            "landmark": [6, 13, 17],
        }
    )

    predictor = MeasurePredictor(ds.landmark_ground_truth)

    test_data["z"] = test_data["position"].apply(
        lambda row: predictor.z_given_x(np.array(row))
    )

    # fig, axes = plt.subplots(1, 3)

    for idx, row in test_data.iterrows():
        # ax: Axes = axes[idx]
        # plot_z_and_landmarks(
        #     row["position"],
        #     row["z"],
        #     ds.landmark_ground_truth,
        #     ax,
        # )
        if plot:
            ax = plot_z_polar(row["position"], row["z"])
            ax.set_title(f"pos={row['position']},mark={row['landmark']}")

        # print out the correct answers to the CLI:
        z = row["z"]
        z_mark = z[z["subject"] == row["landmark"]]
        print(f"PREDICTION for landmark {row['landmark']}, x={row['position']}:")
        print(f" -- bearing : {z_mark['bearing_rad'].item():.3} radians")
        print(f" -- range: {z_mark['range_m'].item():.3} meters")

    # fig.canvas.manager.set_window_title("question_6")
    plt.show()


if __name__ == "__main__":
    main()
