"""
plotting functions for robot data
"""

from matplotlib.figure import Figure
import numpy as np
from matplotlib.axes import Axes
from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbs

from hw0.data import Dataset
from hw0.measure import ZType
from hw0.metrics import abs_trajectory_error, interp2, abs_trajectory_error_rmse


def plot_trajectories_and_particles(
    ds: Dataset,
    traj: pd.DataFrame,
    final_Xt: np.ndarray,
    label: str,
    n_seconds_per_arrow: int = 10,
    traj2: tuple[pd.DataFrame, str] = None,
    final_Xbar_t: np.ndarray | None = None,
) -> None:
    """
    Show a map of the environment, with a given predicted trajectory plotted
    alongside the ground truth trajectory, as well as particles
    from the final frame of the predicted trajectory

    :param ds: full robot dataset
    :param traj: predicted robot trajectory, in the same format as ds.groundtruth
    :param final_particles: particle set from ParticleFilter.get_Xt()
    :param label: descriptive name for this trajectory
    :param traj2: another trajectory for comparison, if desired
    :type traj2: (trajectory, label)
    """
    fig, ax = plot_trajectories_pretty(
        ds, traj, label, n_seconds_per_arrow, traj2=traj2
    )

    tf = traj["time_s"].iloc[-1]
    t0 = traj["time_s"].iloc[0]
    ax.set_title(f"Ground Truth vs. {label} - Particles after {tf - t0:.3f}s")

    # plot particles w/o orientation
    ax.scatter(
        x=final_Xt[:, 0],
        y=final_Xt[:, 1],
        marker=".",
        color="#673aa3",
        s=10,
        # behind everything else
        zorder=0.9,
        label=f"X_t (belief particles) (n={final_Xt.shape[0]})",
    )

    if final_Xbar_t is not None:
        ax.scatter(
            x=final_Xt[:, 0],
            y=final_Xt[:, 1],
            marker=".",
            color="#b9c92d",
            s=100,
            # behind everything else
            zorder=0.8,
            label=f"Xbar_t (proposal particles) (n={final_Xbar_t.shape[0]})",
        )
    ax.legend()


def plot_trajectories_pretty(
    ds: Dataset,
    traj: pd.DataFrame,
    label: str,
    n_seconds_per_arrow: int = 10,
    traj2: tuple[pd.DataFrame, str] = None,
) -> tuple[Figure, Axes]:
    """
    Show a map of the environment, with a given predicted trajectory plotted
    alongside the ground truth trajectory

    :param ds: full robot dataset
    :param traj: predicted robot trajectory, in the same format as ds.groundtruth
    :param label: descriptive name for this trajectory
    :param traj: another trajectory along with its label, if desired.
    no point making this a list cuz if there's more than 2 the plot
    gets impossible to read
    """
    fig = plt.figure()
    ax = fig.subplots()

    ### plot the landmarks as black discs
    centers = []
    for lm in ds.landmarks.itertuples(index=False):
        # these only actually show up if you zoom wayyyyy in. the stdevs are super small.
        oval = patches.Ellipse(
            (lm.x_m, lm.y_m),
            width=lm.x_std_dev * 1000,
            height=lm.y_std_dev * 1000,
            facecolor="black",
            lw=0.5,
        )
        ax.add_patch(oval)

        x, y = oval.center
        centers.append((x, y))
        # text shows up nicely in black boxes
        ax.text(
            x,
            y,
            f"{lm.subject}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ff0055",
            bbox=dict(facecolor="black", edgecolor="#550000", boxstyle="round,pad=0.2"),
        )

    landmark_proxy = patches.Patch(
        facecolor="black", edgecolor="#550000", label="Landmarks"
    )

    ## Set up axes limits
    # they should be consistent, and at least large enough to admit all the landmarks
    # and reasonable trajectories
    xlim = (min(c[0] for c in centers), max(c[0] for c in centers))
    xrange = xlim[1] - xlim[0]
    ylim = (min(c[1] for c in centers), max(c[1] for c in centers))
    yrange = ylim[1] - xlim[0]
    offset = (xrange * 0.8, yrange * 0.8)

    ax.set_xlim(xmin=xlim[0] - offset[0], xmax=xlim[1] + offset[0])
    ax.set_ylim(ymin=ylim[0] - offset[1], ymax=ylim[1] + offset[1])

    ### plot actual trajectories
    _plot_trajectory(
        ax,
        "Groundtruth Traj.",
        ds.ground_truth,
        n_seconds_per_arrow=n_seconds_per_arrow,
        color="#bbbbff",
        start_color="#09ff00",
        end_color="#3232e4",
    )
    _plot_trajectory(
        ax,
        label,
        traj,
        n_seconds_per_arrow=n_seconds_per_arrow,
        color="#443c23",
        # start colors are all same
        start_color="#09ff00",
        end_color="#362828",
    )
    if traj2 is not None:
        _plot_trajectory(
            ax,
            traj2[1],
            traj2[0],
            n_seconds_per_arrow=n_seconds_per_arrow,
            color="#7DA07A8F",
            # same start color as traj1; they will overlay each other.
            start_color="#09ff00",
            end_color="#12411e",
        )

    ## Set up the legend & labels
    ax.plot([], [], " ", label=f"*arrows are {n_seconds_per_arrow}s apart")
    ax.legend(
        handles=[landmark_proxy] + ax.get_legend_handles_labels()[0],
        labels=["Landmarks"] + ax.get_legend_handles_labels()[1],
        loc="center left",
        bbox_to_anchor=(0.8, 0.95),
    )
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Ground Truth vs. {label}")

    return fig, ax


def _plot_trajectory(
    ax: Axes,
    label: str,
    traj: pd.DataFrame,
    color: str,
    start_color: str,
    end_color: str,
    n_seconds_per_arrow: float,
) -> None:
    """
    plots a robot trajectory, with some helpful arrows as it progresses
    """
    ax.plot(
        traj["x_m"],
        traj["y_m"],
        linewidth=0.49,
        color=color,
    )

    ### plot arrows along the path

    # calculate how to distribute our arrows on the trajectory
    # this is based on the data rate of each, so each arrow represents
    # that a certain duration has elapsed
    samples_per_second = len(traj) / (traj.iloc[-1]["time_s"] - traj.iloc[0]["time_s"])
    samples_per_arrow = int(samples_per_second * n_seconds_per_arrow)

    length = 0.2  # in inches
    arrow_locs = traj.iloc[::samples_per_arrow]
    dx = np.cos(arrow_locs["orientation_rad"]) * length
    dy = np.sin(arrow_locs["orientation_rad"]) * length

    ax.quiver(
        arrow_locs["x_m"],
        arrow_locs["y_m"],
        dx,
        dy,
        units="inches",
        angles="xy",
        scale=10,
        scale_units="width",
        color=color,
        width=0.05,
        label=label,
    )

    # add a start vector and end vector
    ax.quiver(
        traj.iloc[0]["x_m"],
        traj.iloc[0]["y_m"],
        np.cos(traj.iloc[0]["orientation_rad"]),
        np.sin(traj.iloc[0]["orientation_rad"]),
        color=start_color,
        label=f"{label} START",
        zorder=2.5,
    )
    ax.quiver(
        traj.iloc[-1]["x_m"],
        traj.iloc[-1]["y_m"],
        np.cos(traj.iloc[-1]["orientation_rad"]),
        np.sin(traj.iloc[-1]["orientation_rad"]),
        color=end_color,
        label=f"{label} END",
        zorder=2.5,
    )


def plot_trajectories_error(
    ds: Dataset, trajectories: dict[str, pd.DataFrame], log=False
) -> None:
    """
    Plot trajectory error over time for multiple trajectories, compared to ground truth

    :param ds: dataset
    :param trajectories: map of descriptive trajectory names to trajectory dataframes,
    in the same format as ds.groundtruth
    """
    # TODO make error function an argument. for now hardcoding ATE.
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    # we're gonna assume here that all trajectories other than groundtruth
    # are the same length+timestamps. if that's not the case, i've done something
    # wrong. so let's check it to catch bugs
    all_lens = [len(traj) for traj in trajectories.values()]
    if not all(l_ == all_lens[0] for l_ in all_lens):
        raise ValueError("Trajectories have different numbers of samples.")

    gt, _ = interp2(ds.ground_truth, next(iter(trajectories.values())))

    for name in trajectories:
        traj = trajectories[name]
        ate = abs_trajectory_error_rmse(gt, traj)

        ax.plot(
            # use relative time so this is more readable
            ate["time_s"] - ate["time_s"].iloc[0],
            ate["ATE_RMS"],
            label=name,
        )

    ax.set_ylabel("Absolute Trajectory Error RMSE (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Trajectory Error vs. Ground Truth")
    if log:
        ax.set_yscale("log")
    ax.legend()


def plot_robot_simple(
    x_all: np.ndarray,
    ax: Axes,
    show_orientations: bool = True,
) -> None:
    """
    Plots the path of the robot given a sequence of states & a time interval between them

    :param x_all: a sequence of t states ordered in time
    :type x_all: np.ndarray [x, y, theta]
    :param dt: time step between each state (or, an array of timestamps)
    :type dt: float | np.ndarray [t_1 - t_0, t_2 - t_1, ... t_n - t_(n-1)]
    :param axes: axes object on which we should plot
    :param display_orientations: show arrows
    """
    # add nice dots for the endpoints
    ax.plot(
        x_all[0, 0],
        x_all[0, 1],
        marker="o",
        markersize=10,
        color="#00bf00",
    )
    ax.plot(
        x_all[-1, 0],
        x_all[-1, 1],
        marker="o",
        markersize=10,
        color="#bf0000",
    )

    # plot each robot location
    ax.plot(
        x_all[:, 0],
        x_all[:, 1],
        linestyle="-",
        marker=".",
        linewidth=1,
        color="black",
    )

    # make arrows for each orientation
    headings = np.column_stack(
        (
            np.cos(x_all[:, 2]),
            np.sin(x_all[:, 2]),
        )
    )

    # TODO make these the same length no matter how far i zoom in?
    # would make it a bit easier to see

    if show_orientations:
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

    ax.grid(True)
    # ax.set_aspect("equal")


def plot_z_and_landmarks(
    x: np.ndarray, z: ZType, landmarks: pd.DataFrame, ax: Axes
) -> None:
    """
    Plot the real landmarks + the landmarks as seen in an observation

    :param x: state (x, y, theta)
    :param z: map of landmark subject #'s to locations
    :type z: ZType
    :param landmarks: landmarks dataframe as in data.py
    :type landmarks: pd.DataFrame
    """
    ax.grid(visible=True)

    # plot the real landmarks
    x_land = landmarks["x_m"].to_numpy()
    y_land = landmarks["y_m"].to_numpy()
    ax.scatter(
        x_land,
        y_land,
        s=200,
        color="skyblue",
        edgecolors="k",
    )

    # label them with their subject #
    for _, row in landmarks[["x_m", "y_m", "subject"]].iterrows():
        ax.text(
            row["x_m"],
            row["y_m"],
            round(row["subject"]),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )

    # add the robot to the plot in the form of an arrow

    length = 1.25  # in inches
    dx = np.cos(x[2]) * length
    dy = np.sin(x[2]) * length

    ax.quiver(
        x[0],
        x[1],
        dx,
        dy,
        units="inches",
        angles="xy",
        scale=10,
        scale_units="width",
        color="red",
        width=0.05,
    )

    # plot the measured landmarks
    z["x_m"] = (z["range_m"] * np.cos(z["bearing_rad"])).to_numpy() + x[0]
    z["y_m"] = (z["range_m"] * np.sin(z["bearing_rad"])).to_numpy() + x[1]
    ax.scatter(
        z["x_m"],
        z["y_m"],
        s=80,
        color="green",
        edgecolors="k",
    )

    # label them with their subject #
    for _, row in z[["x_m", "y_m", "subject"]].iterrows():
        ax.text(
            row["x_m"],
            row["y_m"],
            round(row["subject"]),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )


def plot_z_polar(x: np.ndarray, z: ZType) -> Axes:
    """
    Plot the real landmarks + the landmarks as seen in an observation

    :param x: state (x, y, theta)
    :param z: map of landmark subject #'s to locations
    :type z: ZType
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)  # <-- polar axes
    ax.set_theta_zero_location("N")
    ax.grid(visible=True)

    # add the robot to the plot in the form of an arrow
    # this is in his ref frame so he's always (0,0)

    length = 1.25  # in inches

    ax.quiver(
        0,
        0,
        0,
        length,
        units="inches",
        angles="xy",
        scale=10,
        scale_units="width",
        color="red",
        width=0.05,
    )

    # plot the measured landmarks
    ax.scatter(
        z["bearing_rad"],
        z["range_m"],
        s=200,
        color="green",
        edgecolors="k",
    )

    # label them with their subject #
    for _, row in z.iterrows():
        ax.text(
            row["bearing_rad"],
            row["range_m"],
            round(row["subject"]),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )

    return ax


def plot_weights_stddev(t, weights) -> None:
    """
    plot the standard deviation of particle weights vs. time
    """
    fig = plt.figure()
    ax = fig.subplots()

    ax.scatter(t[:-1], weights[:-1], s=10, color="blue")
    ax.scatter([t[-1]], [weights[-1]], s=10, color="red")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("stddev of particle weights")
    ax.set_title("standard deviation of particle weights vs. time")


def plot_bargraph_error(ds: Dataset, trajectories: dict[str, pd.DataFrame]) -> None:
    """
    Plot total trajectory error for multiple trajectories, compared to ground truth
    Shown as a

    :param ds: dataset
    :param trajectories: map of descriptive trajectory names to trajectory dataframes,
    in the same format as ds.groundtruth
    """
    # TODO make error function an argument. for now hardcoding ATE.
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    # we're gonna assume here that all trajectories other than groundtruth
    # are the same length+timestamps. if that's not the case, i've done something
    # wrong. so let's check it to catch bugs
    all_lens = [len(traj) for traj in trajectories.values()]
    if not all(l_ == all_lens[0] for l_ in all_lens):
        raise ValueError("Trajectories have different numbers of samples.")

    gt, _ = interp2(ds.ground_truth, next(iter(trajectories.values())))

    for name in trajectories:
        traj = trajectories[name]
        ate = abs_trajectory_error_rmse(gt, traj)

        ax.plot(
            # use relative time so this is more readable
            ate["time_s"] - ate["time_s"].iloc[0],
            ate["ATE_RMS"],
            label=name,
        )

    ax.set_ylabel("Absolute Trajectory Error RMSE (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Trajectory Error vs. Ground Truth")
    ax.legend()
