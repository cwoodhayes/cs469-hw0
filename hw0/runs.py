"""
Function for running the filter with a variety of parameters

put this in its own file cuz run.py was getting big.
"""

import numpy as np
import matplotlib.pyplot as plt
from hw0.data import Dataset, Trajectory
from hw0.measure import MeasurementModel
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import GaussianProposalSampler, ParticleFilter
from hw0.runners import ParticleFilterRunner, DEFAULT_RUN_DIR
from hw0 import plot

measurement_cov = np.array(
    [
        [0.5, 0.2],
        [0.2, 0.5],
    ]
)
control_std = 0.5
factors = (10, 1, 0.01, 0.001, 0.0001)
particle_counts = (10, 50, 200)


def run_factors(ds: Dataset) -> None:
    """
    Compare the performance of your motion model (i.e., dead reckoning)
    and your full filter on on the robot dataset (as in step 3).

    This function either generates fresh data or reads it from a file
    """

    runner = ParticleFilterRunner(DEFAULT_RUN_DIR / "run_factors")
    # this is for debugging purposes, to grab only a subset of the points
    # ds = ds.segment_percent(0.0, 0.01, normalize_timestamps=True)
    ds.print_info()

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].iloc[0].to_numpy()

    # now run every combination of the above:
    for i_particle in range(len(particle_counts)):
        for i_meas in range(len(factors)):
            for i_control in range(len(factors)):
                if (i_meas, i_control, i_particle) != (3, 3, 2):
                    continue
                try:
                    run_name = f"run_{i_meas}_{i_control}_{i_particle}"
                    print(f"STARTING RUN: {run_name}")
                    motion = TextbookMotionModel()
                    measure = MeasurementModel(
                        ds.landmarks, cov_matrix=measurement_cov / factors[i_meas]
                    )
                    u_noise = GaussianProposalSampler(
                        stddev=control_std / factors[i_control],
                    )
                    pf_config = ParticleFilter.Config(
                        random_seed=0,
                        n_particles=particle_counts[i_particle],
                    )
                    pf = ParticleFilter(
                        motion,
                        measure,
                        X_0=x_0,
                        config=pf_config,
                        u_noise=u_noise,
                    )
                    runner.run(ds, pf, run_name)
                except Exception as err:
                    if type(err) is KeyboardInterrupt:
                        return
                    else:
                        print(err)
                        continue


def _get_trajectories() -> np.ndarray:
    """
    Get all trajectories generated above. 3-deep nested np.ndarray
    """
    runner = ParticleFilterRunner(DEFAULT_RUN_DIR / "run_factors")

    out = np.empty(
        shape=(len(factors), len(factors), len(particle_counts)), dtype=object
    )

    for i_particle in range(len(particle_counts)):
        for i_meas in range(len(factors)):
            for i_control in range(len(factors)):
                run_name = f"run_{i_meas}_{i_control}_{i_particle}"

                try:
                    traj, pf, traj_gt, name = runner.load(run_name)
                    better_name = (
                        f"{name} - u_f={factors[i_control]}, z_f={factors[i_meas]}"
                    )
                    out[i_meas][i_control][i_particle] = (
                        traj,
                        pf,
                        traj_gt,
                        better_name,
                    )
                except Exception as e:
                    out[i_meas][i_control][i_particle] = None

    return np.array(out)


def plot_all(ds: Dataset) -> None:
    """
    Plot the results of run_factors() above, using the files it generates

    :param ds: Description
    :type ds: Dataset
    """
    trajs = _get_trajectories()

    # plot cumulative error
    ## 50 particles, all noise combos
    trajs_50particles = {item[3]: item[0].df for item in trajs[:, :, 1].flatten()}
    plot.plot_trajectories_error(ds, trajs_50particles)

    ## best noise combo, all particles

    # first 10% time map of the best tuning

    # map of the best tuning, 100% of dataset

    # plot path smoothness (sliding window variance?)

    plt.show()
