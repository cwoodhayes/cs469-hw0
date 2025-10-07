"""
Function for running the filter with a variety of parameters

put this in its own file cuz run.py was getting big.
"""

import numpy as np
from hw0.data import Dataset
from hw0.measure import MeasurementModel
from hw0.motion import TextbookMotionModel
from hw0.particle_filter import GaussianProposalSampler, ParticleFilter
from hw0.runners import ParticleFilterRunner, DEFAULT_RUN_DIR


def run_factors(ds: Dataset) -> None:
    """
    Compare the performance of your motion model (i.e., dead reckoning)
    and your full filter on on the robot dataset (as in step 3).

    This function either generates fresh data or reads it from a file
    """

    measurement_cov = np.array(
        [
            [0.5, 0.2],
            [0.2, 0.5],
        ]
    )
    control_std = 0.5
    factors = (10, 1, 0.01, 0.001, 0.0001)
    particle_counts = (10, 50, 200, 1000)

    runner = ParticleFilterRunner(DEFAULT_RUN_DIR / "run_factors")
    # this is for debugging purposes, to grab only a subset of the points
    # ds = ds.segment_percent(0.0, 0.01, normalize_timestamps=True)
    ds.print_info()

    # grab the initial location from the first ground truth value
    x_0 = ds.ground_truth[["x_m", "y_m", "orientation_rad"]].iloc[0].to_numpy()

    # now run every combination of the above:
    for i_meas in range(len(factors)):
        for i_control in range(len(factors)):
            for i_particle in range(len(particle_counts)):
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
