# Particle Filter for Robot Localization

**Author:** Conor Hayes  
Written for CS/ME469: Machine Learning and Artificial Intelligence for Robotics, Northwestern University ([Prof. Brenna Argall](https://www.argallab.northwestern.edu/people/brenna/))

---

## Overview

A full writeup including analysis, derivations, and discussion of results is available in [writeup.pdf](writeup.pdf). The code in this repo is structured to support the experiments and plots presented in the writeup.

This repo implements a particle filter for mobile robot localization, applied to a real-world wheeled robot dataset from the [UTIAS Multi-Robot Cooperative Localization and Mapping Dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html).

The filter estimates the robot's 2D position and heading over time by combining:
- A **motion model** based on differential-drive kinematics
- A **measurement model** that computes expected range and bearing to known landmarks given LiDAR-based heading data, weighted via Gaussian likelihood
- **Low-variance resampling** for particle set updates (Probabilistic Robotics §4.3)

![Particle filter trajectory](figures/pretty_short.png)
*Fig 8.2 (see writeup) shows the particle filter's estimated trajectory (brown) closely tracking the ground truth (pale purple), while dead reckoning (pale green) diverges. (Only the first 10% of the full trajectory is shown for visual clarity)*

---

## Install

```bash
uv sync
```

Requires Python 3.13+.

---

## Run

```bash
uv run run.py
```

This outputs all plots from the writeup (Questions 2, 3, 6, 7, 8, 9). You'll need to close each plot window before the next one appears.

To regenerate the full grid search across filter hyperparameters (takes ~5 hours):

```bash
python3 run.py --generate
```

Pre-generated trajectory files from prior runs are loaded automatically if present in `data/filter-runs/`.

---

## Data

Place the dataset files in `data/ds1/`:

```
data/ds1/
  ds1_Barcodes.dat
  ds1_Control.dat
  ds1_Groundtruth.dat
  ds1_Landmark_Groundtruth.dat
  ds1_Measurement.dat
```

Files are available on the [UTIAS Multi-Robot Cooperative Localization and Mapping Dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html) site, under MRCLAM_Dataset9, Robot3

---

## Repo Contents

| File | Description |
|------|-------------|
| `run.py` | Entry point; calls one function per question |
| `hw0/motion.py` | Differential-drive motion model |
| `hw0/measure.py` | Range/bearing measurement model with Gaussian likelihood |
| `hw0/particle_filter.py` | Particle filter with low-variance resampling |
| `hw0/runners.py` | Orchestrates filter runs and dead-reckoning on the dataset |
| `hw0/runs.py` | Grid search logic and result plotting |
| `hw0/plot.py` | All plotting functions |
| `hw0/metrics.py` | Trajectory error metrics (RMSE) |
| `hw0/data.py` | Dataset loading and segmentation utilities |

---

## Key Results

- **Dead reckoning** diverges quickly from ground truth due to compounding control uncertainty.
- The **particle filter** tracks the ground truth trajectory closely with well-tuned noise parameters (`u_stddev=0.05`, measurement covariance `[[0.05, 0.02], [0.02, 0.05]]`, 100 particles).
- Performance degrades during periods of missing landmark measurements, as the filter falls back to motion-only propagation.
- Control noise has the largest impact on filter accuracy; measurement noise has a secondary effect.

## Acknowledgements
- Thanks to [Prof. Brenna Argall](https://www.argallab.northwestern.edu/people/brenna/) for providing the assignment for her course *Machine Learning and Artificial Intelligence for Robotics* (CS/ME469) at Northwestern University.
- Thanks to the UTIAS Multi-Robot Cooperative Localization and Mapping Dataset team for providing the curated dataset.