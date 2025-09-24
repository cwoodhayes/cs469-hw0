"""
plotting functions for robot data
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle


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
