import torch
import numpy as np
import os
from typing import Optional
import bisect


class PCAWhitener:
    def __init__(self, out_dim: Optional[int] = None, remove_first_n: int = 1, eps: float = 1e-6):
        self.mu = None
        self.W = None
        self.out_dim = out_dim
        self.remove_first_n = remove_first_n
        self.eps = eps

    @torch.no_grad()
    def fit(self, G: torch.Tensor, max_samples: int = 50000):
        if G.size(0) > max_samples:
            idx = torch.randperm(G.size(0), device=G.device)[:max_samples]
            X = G[idx]
        else:
            X = G

        self.mu = X.mean(dim=0, keepdim=True)
        Xc = X - self.mu
        cov = (Xc.T @ Xc) / (Xc.size(0) - 1.0)

        eigvals, eigvecs = torch.linalg.eigh(cov.float())

        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        start = self.remove_first_n
        if self.out_dim is None:
            end = eigvecs.size(1)
        else:
            end = min(eigvecs.size(1), start + self.out_dim)

        U = eigvecs[:, start:end]
        S_inv_sqrt = (1.0 / torch.sqrt(eigvals[start:end] + self.eps))
        self.W = U * S_inv_sqrt.unsqueeze(0)

    @torch.no_grad()
    def transform(self, G: torch.Tensor) -> torch.Tensor:
        assert self.mu is not None and self.W is not None, "Call fit() first."
        Xc = G - self.mu
        Y = Xc @ self.W
        return l2norm(Y, dim=-1)


@torch.no_grad()
def build_descriptors(tokens: torch.Tensor,
                      whitener: Optional[PCAWhitener] = None,
                      beta: float = 0.5) -> torch.Tensor:
    G = global_mean_pool(tokens)
    G = signed_power_norm(G, beta=beta)
    if whitener is not None:
        G = whitener.transform(G)
    return l2norm(G, dim=-1)


@torch.no_grad()
def cosine_sim_matrix(G: torch.Tensor, block: Optional[int] = None) -> torch.Tensor:
    if block is None:
        return G @ G.T
    N = G.size(0)
    S = torch.empty((N, N), device=G.device, dtype=G.dtype)
    for i in range(0, N, block):
        i2 = min(i + block, N)
        S[i:i2] = G[i:i2] @ G.T
    return S


def l2norm(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def global_mean_pool(tokens: torch.Tensor, chunk_size=64) -> torch.Tensor:
    device = tokens.device
    dtype = tokens.dtype
    N, K, C = tokens.shape
    acc = torch.zeros(N, C, device=device, dtype=dtype)
    count = 0

    for i in range(0, K, chunk_size):
        x = tokens[:, i:i+chunk_size, :].to(dtype)
        x = l2norm(x, dim=-1)
        acc += x.sum(dim=1)
        count += x.shape[1]

    g = acc / count
    return l2norm(g, dim=-1)

@torch.no_grad()
def signed_power_norm(G: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    return l2norm(torch.sign(G) * (G.abs().clamp_min(1e-12).pow(beta)), dim=-1)


@torch.no_grad()
def loop_detection(similarities, indices, similarity_threshold=0.98, neighbor_threshold=100):
    loop_closures = []
    for i in range(len(similarities)):
        for j in range(1, 8):
            neighbor_idx = indices[i, j].item()
            similarity = similarities[i, j].item()

            if similarity > similarity_threshold and abs(i - neighbor_idx) > neighbor_threshold:
                if i < neighbor_idx:
                    loop_closures.append((i, neighbor_idx, similarity))
                else:
                    loop_closures.append((neighbor_idx, i, similarity))

    loop_closures = list(set(loop_closures))
    loop_closures.sort(key=lambda x: x[2], reverse=True)

    loop_closures = _apply_nms_filter(loop_closures, 25)
    loop_closures = _ensure_decending_order(loop_closures)

    return loop_closures


@torch.no_grad()
def _apply_nms_filter(loop_closures, nms_threshold):
    """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
    if not loop_closures or nms_threshold <= 0:
        return loop_closures

    sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
    filtered_loops = []
    suppressed = set()
    
    max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)
    
    for idx1, idx2, sim in sorted_loops:
        if idx1 in suppressed or idx2 in suppressed:
            continue
        
        filtered_loops.append((idx1, idx2, sim))
        
        suppress_range = set()
        
        start1 = max(0, idx1 - nms_threshold)
        end1 = min(idx1 + nms_threshold + 1, idx2) 
        suppress_range.update(range(start1, end1))
        
        start2 = max(idx1 + 1, idx2 - nms_threshold)
        end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
        suppress_range.update(range(start2, end2))
        
        suppressed.update(suppress_range)
    
    return filtered_loops


def _ensure_decending_order(tuples_list):
    return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]


def process_loop_list(chunk_index, loop_list, half_window=10):
    """
    Process loop_list and return chunk indices and frame ranges for each (idx1, idx2) pair.
    chunk_index: List of (begin_idx, end_idx) tuples.
    loop_list: List of (idx1, idx2) tuples.
    half_window: Number of frames to take on each side of center index (default 10).
    Returns list of (chunk_idx1, range1, chunk_idx2, range2) tuples where:
      - chunk_idx1, chunk_idx2: Chunk indices (1-based).
      - range1, range2: Frame range tuples (start, end).
    """
    results = []
    for idx1, idx2 in loop_list:
        try:
            chunk_idx1_0based = find_chunk_index(chunk_index, idx1)
            chunk1 = chunk_index[chunk_idx1_0based]
            range1 = get_frame_range(chunk1, idx1, half_window)
            
            chunk_idx2_0based = find_chunk_index(chunk_index, idx2)
            chunk2 = chunk_index[chunk_idx2_0based]
            range2 = get_frame_range(chunk2, idx2, half_window)
            

            result = (chunk_idx1_0based, range1, chunk_idx2_0based, range2)
            results.append(result)
        except ValueError as e:
            print(f"Skipping pair ({idx1}, {idx2}): {e}")
    return results


def get_frame_range(chunk, idx, half_window=10):
    """
    Calculate the frame range centered at idx with half_window frames on each side within chunk boundaries.
    If near boundaries, take 2 * half_window frames starting from the boundary.
    chunk: (begin_idx, end_idx).
    idx: Center index.
    half_window: Number of frames to take on each side of center index.
    Returns (start, end).
    """
    begin, end = chunk
    window_size = 2 * half_window

    if idx - half_window < begin:
        start = begin
        end_candidate = begin + window_size
        end = min(end, end_candidate)

    elif idx + half_window > end:
        end_candidate = end
        start_candidate = end - window_size
        start = max(begin, start_candidate)

    else:
        start = idx - half_window
        end = idx + half_window
    return (start, end)


def find_chunk_index(chunks, idx):
    """
    Find the 0-based chunk index that contains the given index idx.
    chunks: List of (begin_idx, end_idx).
    idx: The index to search for.
    Returns the 0-based chunk index.
    """
    starts = [chunk[0] for chunk in chunks]
    pos = bisect.bisect_right(starts, idx) - 1
    if pos < 0 or pos >= len(chunks):
        raise ValueError(f"Index {idx} not found in any chunk")
    chunk_begin, chunk_end = chunks[pos]
    if idx < chunk_begin or idx > chunk_end:
        raise ValueError(f"Index {idx} not found in any chunk")
    return pos


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result