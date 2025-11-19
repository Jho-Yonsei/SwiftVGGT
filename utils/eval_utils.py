import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot


mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['lines.linewidth'] = 4


def plot_trajectory(pred_traj, gt_traj=None, output_path=None, 
                    align=True, correct_scale=True, dpi=600):
    assert isinstance(pred_traj, PoseTrajectory3D)

    if gt_traj is not None:
        assert isinstance(gt_traj, PoseTrajectory3D)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    plot_mode = plot.PlotMode.xz
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)

    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray')

    plot.traj(ax, plot_mode, pred_traj, '-', 'green')
    
    fig.tight_layout()

    if output_path:
        fig.savefig(
            os.path.join(output_path, 'trajectory.png'),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=1
        )

    plt.close(fig)