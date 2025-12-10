#!/usr/bin/env python3
"""Compare GT CSV and VINS trajectory CSV (2D) by RMSE."""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import tkinter  # noqa: F401
except ImportError:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


DEFAULT_TRAJ = Path('/root/catkin_ws/src/odometry/outputs/fifthFast_livo.csv')
DEFAULT_GT = Path('/root/catkin_ws/src/odometry/gt/fifth_gt.csv')


def read_xy_points(path: Path) -> np.ndarray:
    points: List[Tuple[float, float]] = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                if len(row) >= 4:
                    # VINS CSV: time, px, py, pz, ...
                    x = float(row[1])
                    y = float(row[2])
                elif len(row) >= 2:
                    x = float(row[0])
                    y = float(row[1])
                else:
                    continue
            except ValueError:
                continue
            points.append((x, y))
    if not points:
        raise RuntimeError(f"No usable data in {path}")
    return np.array(points, dtype=float)


def cumulative_dist(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    dists = np.concatenate(([0.0], np.cumsum(seg)))
    return dists


def resample(points: np.ndarray, target_dists: np.ndarray):
    dists = cumulative_dist(points)
    max_dist = min(target_dists[-1], dists[-1])
    valid = target_dists <= max_dist + 1e-9
    trimmed = target_dists[valid]
    xs = np.interp(trimmed, dists, points[:, 0])
    ys = np.interp(trimmed, dists, points[:, 1])
    return np.column_stack((xs, ys)), valid


def best_fit_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return rotation matrix (2x2) that best aligns `src` to `dst`."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def compute_alignment(gt_path: Path, traj_path: Path):
    gt = read_xy_points(gt_path)
    traj = read_xy_points(traj_path)
    gt_aligned = gt - gt[0]
    traj_shifted = traj - traj[0]
    target_dists = cumulative_dist(gt_aligned)
    traj_resampled, mask = resample(traj_shifted, target_dists)
    gt_subset = gt_aligned[mask]
    R = best_fit_rotation(traj_resampled, gt_subset)
    traj_rotated_subset = (R @ traj_resampled.T).T
    traj_rotated_full = (R @ traj_shifted.T).T
    errors = traj_rotated_subset - gt_subset
    rmse = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
    return gt_aligned, traj_rotated_full, rmse


def main():
    parser = argparse.ArgumentParser(description='Compare GT CSV with LIO trajectory')
    parser.add_argument('--gt', type=Path, default=DEFAULT_GT, help='Ground truth CSV path')
    parser.add_argument('--traj', type=Path, default=DEFAULT_TRAJ, help='Trajectory CSV path')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional path to save comparison figure (default: figures/<traj>_vs_gt.png)')
    args = parser.parse_args()

    gt, traj, rmse = compute_alignment(args.gt, args.traj)
    print(f"RMSE (2D) between {args.gt} and {args.traj}: {rmse:.3f} m")

    figures_dir = Path(__file__).with_name('figures')
    figures_dir.mkdir(exist_ok=True)
    save_path = args.output or figures_dir / f"{args.traj.stem}_vs_gt.png"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gt[:, 0], gt[:, 1], 'o-', label='GT')
    ax.plot(traj[:, 0], traj[:, 1], 'o-', label='Aligned Traj')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'RMSE: {rmse:.3f} m')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved figure to {save_path}")
    if plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()
