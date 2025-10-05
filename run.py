"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib
import signal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hw0.data import Dataset
from hw0.motion import TextbookNoiselessMotionModel
from hw0.particle_filter import ParticleFilter
from hw0.plot import (
    plot_robot_simple,
    plot_z_polar,
    plot_trajectories_pretty,
    plot_trajectories_error,
)
from hw0.measure import MeasurementModel

REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")
    # circle_test(ds)
    # question_2(ds)
    # question_3(ds)
    # question_6(ds)
    question_8b(ds)


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
    plot_robot_simple(states, ax)
    ax.set_title("Q2: Robot path over 5 example commands")
    fig.canvas.manager.set_window_title("question_2")

    plt.show()


def question_3(ds: Dataset) -> None:
    print("!!!!!!!!!!!!!!!!!!! QUESTION 3 !!!!!!!!!!!")

    # this is for debugging purposes, to grab only a subset of the points
    # ds = ds.segment_percent(0, 0.1)

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy()[0]
    # and the first timestamp from the controls
    t_0 = ds.control["time_s"][0]

    # grab the commands
    u_ts = ds.control["time_s"].to_numpy()
    u = ds.control[["forward_velocity_mps", "angular_velocity_radps"]].to_numpy()
    u = u / (10, 1)

    states = [x_0]
    m = TextbookNoiselessMotionModel(x_0, t_0)

    # simulate the robot's motion
    for idx in range(u.shape[0]):
        states.append(m.step_abs_t(u[idx], u_ts[idx]))

    states = np.array(states)

    traj = pd.DataFrame(
        {
            "time_s": ds.control["time_s"],
            "x_m": states[:-1, 0],
            "y_m": states[:-1:, 1],
            "orientation_rad": states[:-1:, 2],
        }
    )
    plot_trajectories_pretty(ds, traj, "Dead-Reckoned Trajectory")
    # plot_trajectories_error(ds, {"Dead-Reckoned Trajectory": traj})
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

    predictor = MeasurementModel(ds.landmarks)

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


def question_8b(ds: Dataset) -> None:
    """
    Compare the performance of your motion model (i.e., dead reckoning)
    and your full filter on on the robot dataset (as in step 3).
    """
    print("!!!!!!!!!!!!!!!!!!! QUESTION 8b !!!!!!!!!!!")

    # this is for debugging purposes, to grab only a subset of the points
    ds = ds.segment_percent(0, 0.1)

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].to_numpy()[0]
    # and the first timestamp from the controls
    t_0 = ds.control["time_s"][0]

    states = []
    motion = TextbookNoiselessMotionModel(x_0, t_0)
    measure = MeasurementModel(ds.landmarks)
    pf = ParticleFilter(motion, measure, X_0=x_0)

    # simulate the robot's motion
    for idx in range(len(ds.control)):
        ctl_series = control = ds.control.iloc[idx]
        ctl_series["forward_velocity_mps"] /= 10
        ctl_series["angular_velocity_radps"] /= 10

        prev_t = pf.get_t()
        curr_t = ctl_series["time_s"]
        meas = ds.measurement_fix[
            (ds.measurement_fix["time_s"] > prev_t)
            & (ds.measurement_fix["time_s"] <= curr_t)
        ]
        x_t = pf.step(control=ctl_series, measurements=meas)
        states.append(x_t)

    states = np.array(states)

    traj = pd.DataFrame(
        {
            "time_s": ds.control["time_s"],
            "x_m": states[:, 0],
            "y_m": states[:, 1],
            "orientation_rad": states[:, 2],
        }
    )
    plot_trajectories_pretty(ds, traj, "Dead-Reckoned Trajectory")
    # plot_trajectories_error(ds, {"Dead-Reckoned Trajectory": traj})
    plt.show()


if __name__ == "__main__":
    main()
