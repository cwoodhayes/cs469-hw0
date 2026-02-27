"""
Runs a variety of filter exercises specified in CS469's Homework 0
"""

import pathlib
import signal
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hw0.data import Dataset, Trajectory
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import GaussianProposalSampler, ParticleFilter
from hw0.plot import (
    plot_robot_simple,
    plot_z_polar,
    plot_trajectories_pretty,
    plot_trajectories_and_particles,
    plot_trajectories_error,
    plot_weights_stddev,
)
from hw0.measure import MeasurementModel
from hw0.integration_tests import circle_test
from hw0.runners import ParticleFilterRunner, dead_reckoner
from hw0.runs import plot_all, run_factors

REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")
    cli = parse_args()
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")
    ds = ds.segment_percent(0.0, 0.12, normalize_timestamps=True)

    # circle_test(ds)
    # question_2(ds)
    # question_3(ds)
    # question_6(ds)
    # question_7(ds)
    question_7(ds)
    # question_8b(ds, write=False)
    # if cli.generate:
    #     run_factors(ds)
    # plot_all(ds)


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="Generate the full dataset of trajectories. Takes several hours.",
    )

    return parser.parse_args()


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
    traj = dead_reckoner(ds)
    plot_trajectories_pretty(ds, traj.df, "Dead-Reckoned Trajectory")
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
            x=np.array(row["position"]), subject=row["landmark"]
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


def question_8b(ds: Dataset, write: bool = False) -> None:
    """
    Compare the performance of your motion model (i.e., dead reckoning)
    and your full filter on on the robot dataset (as in step 3).

    This function either generates fresh data or reads it from a file
    """
    runner = ParticleFilterRunner()

    if write:
        # this is for debugging purposes, to grab only a subset of the points
        # ds = ds.segment_percent(0.0, 0.01, normalize_timestamps=True)
        ds.print_info()

        # grab the initial location from the first ground truth value
        x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].iloc[0].to_numpy()

        motion = TextbookMotionModel()
        measure = MeasurementModel(
            ds.landmarks,
            cov_matrix=np.array(
                [
                    [0.5, 0.2],
                    [0.2, 0.5],
                ]
            )
            / 10,
        )
        u_noise = GaussianProposalSampler(
            stddev=0.05,
        )
        pf_config = ParticleFilter.Config(
            random_seed=0,
            n_particles=100,
        )
        pf = ParticleFilter(
            motion,
            measure,
            X_0=x_0,
            config=pf_config,
            u_noise=u_noise,
        )
        runner.run(ds, pf, "full_test")

    pf_traj, pf, dr_traj, name = runner.load("full_test")
    plot_trajectories_and_particles(
        ds,
        pf_traj.df,
        pf.debug_X_t_last_with_weights,
        "PF Trajectory",
        traj2=(dr_traj.df, "DR Trajectory"),
        final_Xbar_t=pf.debug_Xbar_t_last_with_weights,
    )
    plot_trajectories_error(
        ds,
        {
            "PF Trajectory": pf_traj.df,
            "DR Trajectory": dr_traj.df,
        },
    )
    plt.show()


def question_7(ds: Dataset) -> None:
    """
    implement the full filter
    """
    print("!!!!!!!!!!!!!!!!!!! QUESTION 8b !!!!!!!!!!!")

    # this is for debugging purposes, to grab only a subset of the points
    ds.print_info()

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].iloc[0].to_numpy()

    motion = TextbookMotionModel()
    measure = MeasurementModel(
        ds.landmarks,
        cov_matrix=np.array(
            [
                [0.5, 0.2],
                [0.2, 0.5],
            ]
        )
        / 10,
    )
    u_noise = GaussianProposalSampler(
        stddev=0.05,
    )
    pf_config = ParticleFilter.Config(
        random_seed=0,
        n_particles=100,
    )
    pf = ParticleFilter(
        motion,
        measure,
        X_0=x_0,
        config=pf_config,
        u_noise=u_noise,
    )

    runner = ParticleFilterRunner()
    pf_traj = runner.run(ds, pf, "initial_test", write=False)
    dr_traj = dead_reckoner(ds)

    plot_trajectories_and_particles(
        ds,
        pf_traj.df,
        pf.debug_X_t_last_with_weights,
        "PF Trajectory",
        traj2=(dr_traj.df, "DR Trajectory"),
        final_Xbar_t=pf.debug_Xbar_t_last_with_weights,
    )
    plot_trajectories_error(
        ds,
        {
            "PF Trajectory": pf_traj.df,
            "DR Trajectory": dr_traj.df,
        },
    )
    weights = np.array(pf.debug_weights_stddev)
    plot_weights_stddev(weights[:, 0], weights[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
