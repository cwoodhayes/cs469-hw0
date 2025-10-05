"""
plotting functions for robot data
"""

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from hw0.data import Dataset
from hw0.measure import ZType


def plot_trajectory_pretty(
    ds: Dataset,
    traj: pd.DataFrame,
    label: str,
) -> None:
    """
    Show a map of the environment, with a given predicted trajectory plotted
    alongside the ground truth trajectory

    :param ds: full robot dataset
    :param traj: predicted robot trajectory, in the same format as ds.groundtruth
    :param label: descriptive name for this trajectory
    """
    sns.set_theme(style="darkgrid")

    ds.ground_truth["series"] = "groundtruth"
    traj["series"] = "traj"

    df = pd.concat([ds.ground_truth, traj])

    p = so.Plot(ds.ground_truth, x="x_m", y="y_m")
    p.add(so.Path())
    p.show()

    # sns.lineplot(
    #     data=traj,
    #     sort=False,
    #     x="x_m",
    #     y="y_m",
    #     # hue="series",
    # )


def plot_robot_path(
    x_all: np.ndarray,
    dt: float | np.ndarray,
    ax: Axes,
    show_orientations: bool = True,
    show_points: bool = True,
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
