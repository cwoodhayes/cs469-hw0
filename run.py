"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib
import signal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hw0.data import Dataset
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import GaussianProposalSampler, ParticleFilter
from hw0.plot import (
    plot_robot_simple,
    plot_z_polar,
    plot_trajectories_pretty,
    plot_trajectories_and_particles,
    plot_trajectories_error,
)
from hw0.measure import MeasurementModel
from hw0.integration_tests import circle_test

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


def question_2(ds: Dataset) -> None:
    print("!!!!!!!!!!!!!!!!!!! QUESTION 2 !!!!!!!!!!!")
    m = TextbookMotionModel()

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

    x_prev = np.array([0.0, 0.0, 0.0])
    for idx in range(commands.shape[0]):
        states.append(x_prev)
        x_t = m.step(commands[idx], x_prev, 1.0)
        x_prev = x_t
    states.append(x_prev)

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
    test_data["z_by_landmark"] = test_data.apply(
        lambda row: predictor.z_given_x_by_landmark(
            x=row["position"], subject=row["landmark"]
        ),
        axis=1,
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

        print(f" -- BY_LANDMARK: {row['z_by_landmark'].round(2)}")

    # fig.canvas.manager.set_window_title("question_6")
    plt.show()


def question_8b(ds: Dataset) -> None:
    """
    Compare the performance of your motion model (i.e., dead reckoning)
    and your full filter on on the robot dataset (as in step 3).
    """
    print("!!!!!!!!!!!!!!!!!!! QUESTION 8b !!!!!!!!!!!")

    # this is for debugging purposes, to grab only a subset of the points
    ds = ds.segment_percent(0.0, 0.1, normalize_timestamps=True)
    ds.print_info()

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].iloc[0].to_numpy()
    # and the first timestamp from the controls
    t_0 = ds.control["time_s"].iloc[0]

    states = []
    motion = TextbookMotionModel()
    measure = MeasurementModel(ds.landmarks)
    u_noise = GaussianProposalSampler(stddev=0.005)
    z_noise = GaussianProposalSampler(stddev=0.005)
    pf_config = ParticleFilter.Config(
        random_seed=0,
        n_particles=50,
    )
    pf = ParticleFilter(
        motion,
        measure,
        X_0=x_0,
        config=pf_config,
        u_noise=u_noise,
        z_noise=z_noise,
    )

    # simulate the robot's motion
    # clump together measurements for each control.

    # note that we actually want the measurements _after_ each control (since they are
    # the ones taken while we are executing the command, and which are most relevant
    # to the state after that command)
    # so we insert a dummy control prior to the first one, at the same timestamp
    # as the first measurement
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
    for idx in range(0, len(ds.control) - 1):
        ctl = control.iloc[idx]

        t_ = ctl["time_s"]
        t_next = control["time_s"].iloc[idx + 1]
        dt = t_next - t_

        not_late = ds.measurement_fix["time_s"] <= t_next
        not_early = ds.measurement_fix["time_s"] > t_
        meas = ds.measurement_fix[not_late & not_early]

        x_t = pf.step(control=ctl, measurements=meas, dt=dt)
        states.append(x_t)

    print("Done simulating. Plotting...")
    states = np.array(states)

    traj = pd.DataFrame(
        {
            "time_s": ds.control["time_s"].iloc[1:],
            "x_m": states[:, 0],
            "y_m": states[:, 1],
            "orientation_rad": states[:, 2],
        }
    )
    plot_trajectories_and_particles(ds, traj, pf.get_Xt(), "Dead-Reckoned Trajectory")
    plot_trajectories_error(ds, {"Dead-Reckoned Trajectory": traj})
    plt.show()


if __name__ == "__main__":
    main()
