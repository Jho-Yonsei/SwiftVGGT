import os
import numpy as np

def align_intrinsics_and_depth(Ks, K_ref, depths):
    """
    Ks: (N, 3, 3)
    depths: (N, H, W)
    Align all intrinsics to the first one (K0),
    and rescale depths accordingly.
    """
    N, H, W = depths.shape
    depths_aligned = np.zeros_like(depths)
    Ks_aligned = np.zeros_like(Ks)

    for i in range(N):
        Ki = Ks[i]
        # Scale ratios for fx, fy
        scale_x = K_ref[0, 0] / Ki[0, 0]
        scale_y = K_ref[1, 1] / Ki[1, 1]
        scale_mean = 0.5 * (scale_x + scale_y)

        # Update intrinsic
        K_new = Ki.copy()
        K_new[0, 0] = K_ref[0, 0]
        K_new[1, 1] = K_ref[1, 1]
        K_new[0, 2] = K_ref[0, 2]
        K_new[1, 2] = K_ref[1, 2]

        Ks_aligned[i] = K_new

        # Rescale depth
        depths_aligned[i] = depths[i] * scale_mean

    return Ks_aligned, depths_aligned


def umeyama_sim3(X, Y, with_scale=True, weights=None):
    
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]

    if weights is None:
        w = np.ones(N)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        assert w.shape[0] == N
        w = np.clip(w, 1e-12, None)
    w = w / np.sum(w)

    mu_x = np.sum(X * w[:, None], axis=0)
    mu_y = np.sum(Y * w[:, None], axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    S = (Xc * w[:, None]).T @ Yc
    U, D, Vt = np.linalg.svd(S)
    V = Vt.T

    sgn = np.sign(np.linalg.det(V @ U.T))
    C = np.eye(3)
    C[-1, -1] = sgn
    R = V @ C @ U.T

    if with_scale:
        var_x = np.sum(w * np.sum(Xc**2, axis=1))
        c = np.ones(3); c[-1] = sgn
        s = (D * c).sum() / (var_x + 1e-12)
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return s, R, t


def compute_sim3_ij(S_i, S_j):

    s_i, R_i, T_i = S_i
    s_j, R_j, T_j = S_j

    s_ij = s_j / s_i
    R_ij = R_j @ R_i.T
    T_ij = T_j - s_ij * (R_ij @ T_i)

    return (s_ij, R_ij, T_ij)


def accumulate_sim3_transforms(transforms):
    """
    Accumulate adjacent SIM(3) transforms into transforms from the initial frame to each subsequent frame.
    
    Args:
    transforms: list, each element is a tuple (R, s, t)
        R: 3x3 rotation matrix (np.array)
        s: scale factor (scalar)
        t: 3x1 translation vector (np.array)
    
    Returns:
    Cumulative transforms list, each element is (R_cum, s_cum, t_cum)
        representing the transform from frame 0 to frame k
    """
    if not transforms:
        return []
    
    cumulative_transforms = [transforms[0]]
    
    for i in range(1, len(transforms)):
        s_cum_prev, R_cum_prev, t_cum_prev = cumulative_transforms[i-1]
        s_next, R_next, t_next = transforms[i]
        R_cum_new = R_cum_prev @ R_next
        s_cum_new = s_cum_prev * s_next
        t_cum_new = s_cum_prev * (R_cum_prev @ t_next) + t_cum_prev
        cumulative_transforms.append((s_cum_new, R_cum_new, t_cum_new))
    
    return cumulative_transforms


def apply_sim3_direct(point_maps, s, R, t):
    # point_maps: (b, h, w, 3) -> (b, h, w, 3, 1)
    point_maps_expanded = point_maps[..., np.newaxis]  # (b, h, w, 3, 1)

    # R: (3, 3) -> (b, h, w, 3, 1) = (3, 3) @ (3, 1)
    rotated = np.matmul(R, point_maps_expanded)  # (b, h, w, 3, 1)
    rotated = rotated.squeeze(-1)  # (b, h, w, 3)
    transformed = s * rotated + t  # (b, h, w, 3)

    return transformed


def confident_pointcloud(points, colors, confs, conf_threshold, sample_ratio=1.0, batch_size=1000000):
    """
    - points: np.ndarray,  (b, H, W, 3) / (N, 3)
    - colors: np.ndarray,  (b, H, W, 3) / (N, 3)
    - confs: np.ndarray,  (b, H, W) / (N,)
    - output_path: str
    - conf_threshold: float,
    - sample_ratio: float (0 < sample_ratio <= 1.0)
    - batch_size: int
    """
    if points.ndim == 2:
        b = 1
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]
    elif points.ndim == 4:
        b = points.shape[0]
    else:
        raise ValueError("Unsupported points dimension. Must be 2 (N,3) or 4 (b,H,W,3)")
    
    total_valid = 0
    for i in range(b):
        cfs = confs[i].reshape(-1)
        total_valid += np.count_nonzero((cfs >= conf_threshold) & (cfs > 1e-5))
    
    num_samples = int(total_valid * sample_ratio) if sample_ratio < 1.0 else total_valid
    
    reservoir_pts = np.zeros((num_samples, 3), dtype=np.float32)
    reservoir_clr = np.zeros((num_samples, 3), dtype=np.uint8)
    count = 0
    
    for i in range(b):
        pts = points[i].reshape(-1, 3).astype(np.float32)
        cls = colors[i].reshape(-1, 3).astype(np.uint8)
        cfs = confs[i].reshape(-1).astype(np.float32)
        
        mask = (cfs >= conf_threshold) & (cfs > 1e-5)
        valid_pts = pts[mask]
        valid_cls = cls[mask]
        n_valid = len(valid_pts)
        
        if count < num_samples:
            fill_count = min(num_samples - count, n_valid)
            
            reservoir_pts[count:count+fill_count] = valid_pts[:fill_count]
            reservoir_clr[count:count+fill_count] = valid_cls[:fill_count]
            count += fill_count
            
            if fill_count < n_valid:
                remaining_pts = valid_pts[fill_count:]
                remaining_cls = valid_cls[fill_count:]
                
                count, reservoir_pts, reservoir_clr = optimized_vectorized_reservoir_sampling(
                    remaining_pts, remaining_cls, count, reservoir_pts, reservoir_clr
                )
        else:
            count, reservoir_pts, reservoir_clr = optimized_vectorized_reservoir_sampling(
                valid_pts, valid_cls, count, reservoir_pts, reservoir_clr
            )
    
    return reservoir_pts, reservoir_clr


def optimized_vectorized_reservoir_sampling(
    new_points: np.ndarray,
    new_colors: np.ndarray,
    current_count: int, 
    reservoir_points: np.ndarray,
    reservoir_colors: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Optimized vectorized reservoir sampling with batch probability calculations.
    
    This maintains mathematical correctness while improving performance through
    vectorized operations where possible.
    
    Args:
        new_points: New point coordinates to consider, shape (M, 3)
        new_colors: New point colors to consider, shape (M, 3)
        current_count: Number of elements seen so far  
        reservoir_points: Current reservoir of sampled points, shape (K, 3)
        reservoir_colors: Current reservoir of sampled colors, shape (K, 3)
        
    Returns:
        Tuple of (updated_count, updated_reservoir_points, updated_reservoir_colors)
    """
    random_gen = np.random
        
    reservoir_size = len(reservoir_points)
    num_new_points = len(new_points)
    
    if num_new_points == 0:
        return current_count, reservoir_points, reservoir_colors
    
    # Calculate sequential indices for each new point
    point_indices = np.arange(current_count + 1, current_count + num_new_points + 1)
    
    # Generate random numbers for each point
    random_values = random_gen.randint(0, point_indices, size=num_new_points)
    
    # Determine which points should replace reservoir elements
    replacement_mask = random_values < reservoir_size
    replacement_positions = random_values[replacement_mask]
    
    # Apply replacements
    if np.any(replacement_mask):
        points_to_replace = new_points[replacement_mask]
        colors_to_replace = new_colors[replacement_mask]
        
        reservoir_points[replacement_positions] = points_to_replace
        reservoir_colors[replacement_positions] = colors_to_replace
    
    return current_count + num_new_points, reservoir_points, reservoir_colors


def save_camera_poses(all_camera_extrinsics, all_camera_intrinsics, sim3_list, output_path, num_imgs):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        
        all_poses = [None] * num_imgs
        all_intrinsics = [None] * num_imgs
        
        first_chunk_range, first_chunk_extrinsics = all_camera_extrinsics[0]
        _, first_chunk_intrinsics = all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i] 
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(all_camera_extrinsics)):
            chunk_range, chunk_extrinsics = all_camera_extrinsics[chunk_idx]
            _, chunk_intrinsics = all_camera_intrinsics[chunk_idx]
            s, R, t = sim3_list[chunk_idx-1]   # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.
            
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i]
        
        poses_path = os.path.join(output_path, "camera_poses.txt")
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()[:12]
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        intrinsics_path = os.path.join(output_path, 'intrinsic.txt')
        with open(intrinsics_path, 'w') as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f'{fx} {fy} {cx} {cy}\n')