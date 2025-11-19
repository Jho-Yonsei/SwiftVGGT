import os
import argparse
from pathlib import Path

import torch
import numpy as np

import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from utils.eval_utils import plot_trajectory

from swiftvggt import SwiftVGGT

def main():
    parser = argparse.ArgumentParser(description="SwiftVGGT Inference")
    parser.add_argument("--image_dir", type=Path, required=True, help="Image directory containing images")
    parser.add_argument("--gt_pose_path", type=Path)
    parser.add_argument("--ckpt", type=Path, default='./ckpt/model_tracker_fixed_e20.pt')
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--save_points", action='store_true')
    # Loop Detection
    parser.add_argument("--chunk_size", type=int, default=75)
    parser.add_argument("--overlap_size", type=int, default=30)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--similarity_threshold", type=float, default=0.98)
    parser.add_argument("--neighbor_threshold", type=int, default=100)
    parser.add_argument("--pooling_batch_size", type=int, default=64)
    parser.add_argument("--pca_out_dim", type=int, default=512)
    parser.add_argument("--pca_remove_first_n", type=int, default=1)
    parser.add_argument("--signed_power_beta", type=float, default=0.5)
    # Temporal Chunks
    parser.add_argument("--depth_diff_threshold", type=float, default=0.2)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    # Loop Optimization
    parser.add_argument("--loop_max_iteration", type=int, default=1000)
    parser.add_argument("--loop_lambda_init", type=float, default=1e-6)
    # Point Cloud
    parser.add_argument("--point_conf_threshold", type=float, default=0.75)
    parser.add_argument("--sampling_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    
    print("Output Path:", args.output_path)
    os.makedirs(args.output_path, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Using device: {device}")
    print(f"Using dtype : {dtype}")
    
    model = SwiftVGGT(args, device, dtype)
    model.run()
    
    if args.gt_pose_path:
        traj_gt = args.gt_pose_path
        traj_est = os.path.join(args.output_path, 'camera_poses.txt')
        
        poses_gt = file_interface.read_kitti_poses_file(traj_gt)
        poses_est = file_interface.read_kitti_poses_file(traj_est)
        
        traj_gt = PoseTrajectory3D(poses_se3=poses_gt.poses_se3,
                                    timestamps=np.arange(poses_gt.num_poses, dtype=np.float64))
        
        traj_est = PoseTrajectory3D(poses_se3=poses_est.poses_se3,
                                    timestamps=np.arange(poses_est.num_poses, dtype=np.float64))
        
        traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
        
        plot_trajectory(traj_est, traj_gt, args.output_path, align=True, correct_scale=True)
        
        result = main_ape.ape(traj_gt, traj_est, est_name='traj', pose_relation=PoseRelation.full_transformation, align=True, correct_scale=True)
        with open(os.path.join(args.output_path, 'result.txt'), 'w') as f:
            print(result, file=f)


if __name__ == "__main__":
    main()