import torch
import numpy as np
import trimesh
import scipy

import os
from glob import glob
from tqdm import tqdm
import time

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import load_images_rgb, get_vgg_input_imgs
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from utils.chunk_utils import *
from utils.loop_utils import *
from utils.loop_optimizer import *


class SwiftVGGT:
    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype
        
        self.load_model(args.ckpt)
        self.load_images(args.image_dir)
        self.num_chunks, self.chunk_list = self.get_idx_list(args.chunk_size, args.overlap_size)
        
        self.loop_optimizer = Sim3LoopOptimizer(args)
        
        self.camera_extrinsics = []
        self.camera_intrinsics = []
        self.temporal_chunk_list = []
        self.sim3_list = []
        self.loop_prediction_list = []
        self.loop_sim3_list = []
        
        
    @torch.no_grad()
    def run_loop_detection(self, args):
        num_chunks_loop, chunk_list_loop = self.get_idx_list(chunk_size=64, overlap_size=0)
        patch_tokens_list = []
        print("🔧 Images to DINO Patch Tokens")
        for batch_idx, (start, end) in enumerate(chunk_list_loop):
            print(f"\tProcessing batch {batch_idx + 1:3d}/{num_chunks_loop}: images {start:4d} to {end - 1:4d}", end='\r', flush=True)
            
            patch_tokens = self.model.aggregator.forward_DINOv2(self.vgg_input[start: end])
            patch_tokens_list.append(patch_tokens)
            del patch_tokens
        print('\n')
        
        patch_tokens = torch.cat(patch_tokens_list, dim=0)
        del patch_tokens_list
        
        start = time.time()
        
        G = self.get_descriptors(patch_tokens, args)
        S_cos = cosine_sim_matrix(G)
        similarities, indices = S_cos.topk(k=args.topk)
        
        loop_closures = loop_detection(similarities, indices, 
                                       similarity_threshold=args.similarity_threshold,
                                       neighbor_threshold=args.neighbor_threshold)
        del similarities, indices, S_cos
        torch.cuda.empty_cache()
        
        loop_list = [(idx1, idx2) for idx1, idx2, _ in loop_closures]
        loop_results = process_loop_list(self.chunk_list, loop_list, half_window=10)
        loop_results = remove_duplicates(loop_results)
        
        end = time.time()
        
        return patch_tokens, loop_results
    
        
    def get_descriptors(self, patch_tokens, args):
        G0 = global_mean_pool(patch_tokens.float(), chunk_size=args.pooling_batch_size)
        
        whitener = PCAWhitener(out_dim=args.pca_out_dim, remove_first_n=args.pca_remove_first_n)
        whitener.fit(G0)
        
        G = build_descriptors(patch_tokens.float(), whitener=whitener, beta=args.signed_power_beta)
        
        return G
            
        
    def get_idx_list(self, chunk_size, overlap_size):
        step_size = chunk_size - overlap_size
        idx_list = []
        for start_idx in range(0, len(self.image_paths), step_size):
            end_idx = start_idx + chunk_size
            if end_idx > len(self.image_paths):
                end_idx = len(self.image_paths)
            idx_list.append([start_idx, end_idx])
            if end_idx == len(self.image_paths):
                break
        return len(idx_list), idx_list
        
        
    def load_images(self, image_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
        images = load_images_rgb(self.image_paths)
        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        self.vgg_input = vgg_input.to(device=self.device)
        self.model.update_patch_dimensions(patch_width, patch_height)
        
        print(f"📐 Image patch dimensions: {patch_width}x{patch_height}", end='\n\n')
        
        
    def load_model(self, ckpt='./ckpt/model_tracker_fixed_e20.pt'):
        print(f"🔄 Loading model: '{ckpt}'")
        model = VGGT(merging=0, vis_attn_map=False)
        ckpt = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model = model.cuda().eval()
        self.model = model.to(self.dtype)
        print("✅ Model loaded")
        
    
    def run_single_temporal_chunk(self, patch_tokens_chunk, images_chunk, chunk_idx, start_idx, end_idx):
        tokens_chunk, patch_start_idx = self.model.aggregator.forward_aggregator(patch_tokens_chunk, *images_chunk.shape[-2:])
        cam_tokens_chunk, patch_tokens_chunk = tokens_chunk[-1][:, :, 0:1, :], torch.stack(tokens_chunk)[:, :, :, 5:, :]
        prediction_chunk = self.model.run_heads(cam_tokens_chunk, patch_tokens_chunk, images_chunk, patch_start_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction_chunk['pose_enc'], images_chunk.shape[-2:])
        depth_map, depth_conf = prediction_chunk['depth'], prediction_chunk['depth_conf']
        del prediction_chunk
        
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()
        depth_map = depth_map.squeeze(0).cpu().float().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().float().numpy()
        
        if chunk_idx == 0:
            self.intrinsic_ref = intrinsic[0]
            
        intrinsic, depth_map = align_intrinsics_and_depth(intrinsic, self.intrinsic_ref, depth_map[..., 0])
        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)
        
        if chunk_idx != 0:
            depth_map_prev = self.temporal_chunk_list[-1][1]['depth_map']
            depth_conf_prev = self.temporal_chunk_list[-1][1]['depth_conf']
            
            depth_map_prev_overlap = depth_map_prev[self.args.chunk_size - self.args.overlap_size:]
            depth_conf_prev_overlap = depth_conf_prev[self.args.chunk_size - self.args.overlap_size:]
            
            depth_map_cur_overlap = depth_map[:self.args.overlap_size]
            depth_conf_cur_overlap = depth_conf[:self.args.overlap_size]
            
            mask = (np.abs(depth_map_prev_overlap - depth_map_cur_overlap) < self.args.depth_diff_threshold) \
                * (depth_conf_prev_overlap > depth_conf_prev_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold) \
                * (depth_conf_cur_overlap > depth_conf_cur_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold)
                
            if mask.sum() < 20000:
                mask = (np.abs(depth_map_prev_overlap - depth_map_cur_overlap) < self.args.depth_diff_threshold + 0.05) \
                    * (depth_conf_prev_overlap > depth_conf_prev_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold) \
                    * (depth_conf_cur_overlap > depth_conf_cur_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold)
                    
            points_3d_prev = self.temporal_chunk_list[-1][1]['points_3d']
            
            points_3d_prev_overlap = points_3d_prev[self.args.chunk_size - self.args.overlap_size:]
            points_3d_cur_overlap = points_3d[:self.args.overlap_size]
            
            points_3d_prev_overlap = points_3d_prev_overlap[mask]
            points_3d_cur_overlap = points_3d_cur_overlap[mask]
            
            s, R, t = umeyama_sim3(points_3d_cur_overlap, points_3d_prev_overlap)
            
            self.sim3_list.append((s, R, t))
            
        prediction_chunk = {'extrinsic': extrinsic, 'intrinsic': intrinsic, 'depth_map': depth_map, 
                            'depth_conf': depth_conf, 'points_3d': points_3d}
        
        self.camera_extrinsics.append((self.chunk_list[chunk_idx], extrinsic))
        self.camera_intrinsics.append((self.chunk_list[chunk_idx], intrinsic))
        self.temporal_chunk_list.append(((start_idx, end_idx), prediction_chunk))
        
    
    def run_temporal_chunks(self, patch_tokens):
        print("🔧 Process Temporal Chunks")
        
        for chunk_idx in range(self.num_chunks):
            start_idx, end_idx = self.chunk_list[chunk_idx]
            print(f"\tProcessing chunk {chunk_idx + 1:3d}/{self.num_chunks}: images {start_idx:4d} to {end_idx - 1:4d}", end='\r', flush=True)
            
            patch_tokens_chunk = patch_tokens[start_idx: end_idx]
            images_chunk = self.vgg_input[start_idx: end_idx]
            
            self.run_single_temporal_chunk(patch_tokens_chunk, images_chunk, chunk_idx, start_idx, end_idx)
        print('\n')
            
            
    def run_single_loop_centric_chunks(self, patch_tokens_chunk, images_chunk, item):
        tokens_chunk, patch_start_idx = self.model.aggregator.forward_aggregator(patch_tokens_chunk, *images_chunk.shape[-2:])
        cam_tokens_chunk, patch_tokens_chunk = tokens_chunk[-1][:, :, 0:1, :], torch.stack(tokens_chunk)[:, :, :, 5:, :]
        prediction_chunk = self.model.run_heads(cam_tokens_chunk, patch_tokens_chunk, images_chunk, patch_start_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction_chunk['pose_enc'], images_chunk.shape[-2:])
        depth_map, depth_conf = prediction_chunk['depth'], prediction_chunk['depth_conf']
        del prediction_chunk
        
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()
        depth_map = depth_map.squeeze(0).cpu().float().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().float().numpy()
        
        intrinsic, depth_map = align_intrinsics_and_depth(intrinsic, self.intrinsic_ref, depth_map[..., 0])
        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)
        
        prediction_chunk = {'extrinsic': extrinsic, 'intrinsic': intrinsic, 'depth_map': depth_map, 
                            'depth_conf': depth_conf, 'points_3d': points_3d}
        
        self.loop_prediction_list.append((item, prediction_chunk))
        
        
    def run_loop_centric_chunks(self, patch_tokens, loop_results):
        print("🔧 Process Loop-Centric Chunks")
        for idx, item in enumerate(loop_results):
            start1, end1 = item[1]
            start2, end2 = item[3]
            print(f"\tProcessing chunk {idx + 1:3d}/{len(loop_results)}: images {start1:4d} to {end1 - 1:4d} & {start2:4d} to {end2 - 1:4d}", end='\r', flush=True)
            
            patch_tokens_chunk = torch.cat([patch_tokens[start1: end1], patch_tokens[start2: end2]], dim=0)
            images_chunk = torch.cat([self.vgg_input[start1: end1], self.vgg_input[start2: end2]], dim=0)
            
            self.run_single_loop_centric_chunks(patch_tokens_chunk, images_chunk, item)
        print('\n')
    
    
    def get_sim3_for_single_loop(self, chunk_idx, chunk_range, predictions, target):
        if target == False:
            points_3d_chunk = predictions['points_3d'][:chunk_range[1] - chunk_range[0]]
            depth_map_chunk = predictions['depth_map'][:chunk_range[1] - chunk_range[0]]
            conf_chunk = predictions['depth_conf'][:chunk_range[1] - chunk_range[0]]
            chunk_begin = chunk_range[0] - self.chunk_list[chunk_idx][0]
            chunk_end = chunk_begin + chunk_range[1] - chunk_range[0]
        else:
            points_3d_chunk = predictions['points_3d'][-chunk_range[1] + chunk_range[0]:]
            depth_map_chunk = predictions['depth_map'][-chunk_range[1] + chunk_range[0]:]
            conf_chunk = predictions['depth_conf'][-chunk_range[1] + chunk_range[0]:]
            chunk_begin = chunk_range[0] - self.chunk_list[chunk_idx][0]
            chunk_end = chunk_begin + chunk_range[1] - chunk_range[0]
        
        chunk_data = self.temporal_chunk_list[chunk_idx][1]

        points_3d = chunk_data['points_3d'][chunk_begin: chunk_end]
        depth_map = chunk_data['depth_map'][chunk_begin: chunk_end]
        conf = chunk_data['depth_conf'][chunk_begin: chunk_end]
        
        mask = (np.abs(depth_map_chunk - depth_map) < self.args.depth_diff_threshold * 2) \
            * (conf_chunk > conf_chunk.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold) \
            * (conf > conf.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold)
        
        s, R, t = umeyama_sim3(points_3d_chunk[mask], points_3d[mask])
        
        return s, R, t
    
    
    def get_sim3_for_loops(self):
        print("🔧 Estimate Sim(3) for Loop")
        for idx, item in enumerate(self.loop_prediction_list):
            print(f"\tSim(3) Align {idx + 1:3d}/{len(self.loop_prediction_list)}", end='\r', flush=True)
            chunk_idx_i, chunk_range_i = item[0][0], item[0][1]
            chunk_idx_j, chunk_range_j = item[0][2], item[0][3]
            predictions = item[1]
            
            s_i, R_i, t_i = self.get_sim3_for_single_loop(chunk_idx_i, chunk_range_i, predictions, target=False)
            s_j, R_j, t_j = self.get_sim3_for_single_loop(chunk_idx_j, chunk_range_j, predictions, target=True)
            
            s_ij, R_ij, t_ij = compute_sim3_ij((s_i, R_i, t_i), (s_j, R_j, t_j))
            
            self.loop_sim3_list.append((chunk_idx_i, chunk_idx_j, (s_ij, R_ij, t_ij)))
        print('\n')
        
    
    def align_chunks(self, sim3_list):
        points_list = []
        colors_list = []
        print("🔧 Align Temporal Chunks")
        for idx, (start, end) in enumerate(self.chunk_list):
            chunk_data = self.temporal_chunk_list[idx][1]
            if idx != 0:
                print(f"\tAligning {idx:3d} -> {idx - 1:3d} (Total {self.num_chunks - 1})", end='\r', flush=True)
                s, R, t = sim3_list[idx - 1]
                chunk_data['points_3d'] = apply_sim3_direct(chunk_data['points_3d'], s, R, t)
            
            points = chunk_data['points_3d'].reshape(-1, 3)
            colors = (self.vgg_input[start: end].cpu().detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = chunk_data['depth_conf'].reshape(-1)
            conf_threshold = np.mean(confs) * self.args.point_conf_threshold
            
            points, colors = confident_pointcloud(points=points, colors=colors, confs=confs, conf_threshold=conf_threshold, sample_ratio=self.args.sampling_ratio)
            points_list.append(points)
            colors_list.append(colors)
        
        points_3d = np.concatenate(points_list)
        colors = np.concatenate(colors_list)
        print('\n')
        
        return points_3d, colors

    
    @torch.no_grad()
    def run(self):
        start = time.time()
        
        patch_tokens, loop_results = self.run_loop_detection(self.args)
        
        self.run_temporal_chunks(patch_tokens)
        self.run_loop_centric_chunks(patch_tokens, loop_results)
        self.get_sim3_for_loops()
        
        sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        sim3_list = accumulate_sim3_transforms(sim3_list)
        
        points_3d, colors = self.align_chunks(sim3_list)
        
        end = time.time()
        print("🔦 Elapsed Time for Whole Process: ", end - start)
        
        save_camera_poses(self.camera_extrinsics, self.camera_intrinsics, sim3_list, self.args.output_path, num_imgs=self.vgg_input.shape[0])
        if self.args.save_points:
            trimesh.PointCloud(points_3d, colors=colors).export(os.path.join(self.args.output_path, "./points_3d.ply"))
            
            