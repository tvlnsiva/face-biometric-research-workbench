"""Milestone P3 (Section 3F) — Vector Retrieval & Approximate Nearest Neighbor Research Indexes.

Implements exact and approximate 1:N vector search algorithms from first principles using the
existing validated `.venv` (`numpy`, `scikit-learn`, `torch`) without introducing external
database infrastructure or modifying `.venv`:

  1. `ExactFlatIndex`: Exact inner-product (`ip` / cosine on unit vectors) and squared-L2 (`l2`)
     search on CPU (NumPy BLAS SGEMM) and GPU (`torch.matmul` in FP32 or FP16).
  2. `IVFFlatIndex`: Coarse Voronoi partitioning via spherical/Euclidean `MiniBatchKMeans`
     (`nlist` cells) with configurable multi-probe search (`nprobe`).
  3. `IVFPQIndex`: Coarse IVF residual quantization + Product Quantization (`M` subspaces x `256`
     8-bit centroids = `M` bytes per vector, e.g., `32 B` vs. `2,048 B` FP32 = `64x` compression)
     accelerated via vectorized Asymmetric Distance Computation (`ADC`) lookup tables.
  4. `HNSWIndex`: Multi-layer Hierarchical Navigable Small World proximity graph (Malkov &
     Yashunin, IEEE TPAMI 2020) with configurable `M`, `ef_construction`, and `ef_search`.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def _ensure_2d_f32(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D embedding array; got shape {out.shape}")
    return np.ascontiguousarray(out)


class ExactFlatIndex:
    """Exact exhaustive brute-force vector index (CPU BLAS or PyTorch CUDA FP32/FP16)."""

    def __init__(self, dim: int, metric: str = "ip", backend: str = "numpy_cpu") -> None:
        if metric not in {"ip", "l2"}:
            raise ValueError(f"Unsupported metric='{metric}'; expected 'ip' or 'l2'")
        if backend not in {"numpy_cpu", "torch_cuda_fp32", "torch_cuda_fp16"}:
            raise ValueError(f"Unsupported backend='{backend}'")
        self.dim = int(dim)
        self.metric = metric
        self.backend = backend
        self._vectors: np.ndarray = np.empty((0, self.dim), dtype=np.float32)
        self._norms_sq: np.ndarray = np.empty((0,), dtype=np.float32)
        self._gpu_vectors: torch.Tensor | None = None
        self._gpu_norms_sq: torch.Tensor | None = None

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def memory_bytes(self) -> int:
        if self._gpu_vectors is not None:
            b = int(self._gpu_vectors.numel() * self._gpu_vectors.element_size())
            if self._gpu_norms_sq is not None:
                b += int(self._gpu_norms_sq.numel() * self._gpu_norms_sq.element_size())
            return b
        return int(self._vectors.nbytes + self._norms_sq.nbytes)

    def add(self, vectors: np.ndarray) -> None:
        mat = _ensure_2d_f32(vectors)
        if mat.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {mat.shape[1]}")
        self._vectors = mat.copy()
        self._norms_sq = np.sum(self._vectors * self._vectors, axis=1).astype(np.float32)

        if self.backend.startswith("torch_cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for ExactFlatIndex but torch.cuda.is_available() is False")
            dtype = torch.float16 if self.backend.endswith("fp16") else torch.float32
            self._gpu_vectors = torch.from_numpy(self._vectors).to(device="cuda", dtype=dtype)
            if self.metric == "l2":
                self._gpu_norms_sq = torch.from_numpy(self._norms_sq).to(device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()

    def search(self, queries: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = _ensure_2d_f32(queries)
        k = min(int(top_k), self.ntotal)
        if k <= 0:
            raise ValueError("Index is empty or top_k <= 0")

        if self.backend.startswith("torch_cuda") and self._gpu_vectors is not None:
            dtype = self._gpu_vectors.dtype
            with torch.no_grad():
                qt = torch.from_numpy(q).to(device="cuda", dtype=dtype)
                dots = torch.matmul(qt, self._gpu_vectors.T).float()
                if self.metric == "ip":
                    vals, idx = torch.topk(dots, k=k, dim=1, largest=True, sorted=True)
                else:
                    q_sq = torch.sum(qt.float() ** 2, dim=1, keepdim=True)
                    dists = torch.clamp(q_sq + self._gpu_norms_sq.unsqueeze(0) - 2.0 * dots, min=0.0)  # type: ignore[union-attr]
                    vals_neg, idx = torch.topk(-dists, k=k, dim=1, largest=True, sorted=True)
                    vals = -vals_neg
                torch.cuda.synchronize()
                return vals.cpu().numpy(), idx.cpu().numpy().astype(np.int64)

        dots_np = q @ self._vectors.T  # (Q, N)
        if self.metric == "ip":
            scores = dots_np
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(q.shape[0])[:, None]
            sub_scores = scores[row_idx, part]
            order = np.argsort(-sub_scores, axis=1)
            top_idx = part[row_idx, order]
            top_vals = sub_scores[row_idx, order]
            return top_vals.astype(np.float32), top_idx.astype(np.int64)

        q_sq_np = np.sum(q * q, axis=1, keepdims=True)
        dists_np = np.maximum(0.0, q_sq_np + self._norms_sq[None, :] - 2.0 * dots_np)
        part = np.argpartition(dists_np, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(q.shape[0])[:, None]
        sub_d = dists_np[row_idx, part]
        order = np.argsort(sub_d, axis=1)
        top_idx = part[row_idx, order]
        top_vals = sub_d[row_idx, order]
        return top_vals.astype(np.float32), top_idx.astype(np.int64)


class IVFFlatIndex:
    """Inverted File Index (`IVF-Flat`) with coarse k-means Voronoi cells and `nprobe` search."""

    def __init__(self, dim: int, nlist: int = 256, seed: int = 42) -> None:
        self.dim = int(dim)
        self.nlist = int(nlist)
        self.seed = int(seed)
        self.centroids: np.ndarray | None = None
        self.invlists_vecs: list[np.ndarray] = []
        self.invlists_ids: list[np.ndarray] = []
        self._ntotal = 0

    @property
    def ntotal(self) -> int:
        return self._ntotal

    def memory_bytes(self) -> int:
        total = 0 if self.centroids is None else int(self.centroids.nbytes)
        for v, i in zip(self.invlists_vecs, self.invlists_ids):
            total += int(v.nbytes + i.nbytes)
        return total

    def train_and_add(self, vectors: np.ndarray, train_sample_max: int = 25000) -> None:
        mat = _ensure_2d_f32(vectors)
        n, d = mat.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        eff_nlist = min(self.nlist, n)
        self.nlist = eff_nlist

        rng = np.random.default_rng(self.seed)
        if n > train_sample_max:
            sub_idx = rng.choice(n, size=train_sample_max, replace=False)
            train_data = mat[sub_idx]
        else:
            train_data = mat

        km = MiniBatchKMeans(
            n_clusters=eff_nlist,
            batch_size=2048,
            max_iter=60,
            n_init=1,
            random_state=self.seed,
        )
        km.fit(train_data)
        raw_c = km.cluster_centers_.astype(np.float32)
        c_norms = np.maximum(np.linalg.norm(raw_c, axis=1, keepdims=True), 1e-12)
        self.centroids = np.ascontiguousarray(raw_c / c_norms, dtype=np.float32)

        # Assign every gallery vector to its highest inner-product centroid
        assign = np.empty(n, dtype=np.int32)
        chunk = 8192
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            assign[s:e] = np.argmax(mat[s:e] @ self.centroids.T, axis=1)

        order = np.argsort(assign, kind="stable")
        sorted_assign = assign[order]
        counts = np.bincount(sorted_assign, minlength=eff_nlist)
        splits = np.cumsum(counts)[:-1]
        id_groups = np.split(order.astype(np.int64), splits)

        self.invlists_ids = [np.ascontiguousarray(g) for g in id_groups]
        self.invlists_vecs = [np.ascontiguousarray(mat[g]) for g in id_groups]
        self._ntotal = n

    def search(self, queries: np.ndarray, top_k: int = 10, nprobe: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self.centroids is None:
            raise RuntimeError("IVFFlatIndex is not trained.")
        q = _ensure_2d_f32(queries)
        nq = q.shape[0]
        k = min(int(top_k), self._ntotal)
        eff_probe = min(max(1, int(nprobe)), self.nlist)

        coarse_sims = q @ self.centroids.T  # (Q, nlist)
        if eff_probe < self.nlist:
            probe_cells = np.argpartition(-coarse_sims, kth=eff_probe - 1, axis=1)[:, :eff_probe]
        else:
            probe_cells = np.tile(np.arange(self.nlist), (nq, 1))

        out_scores = np.full((nq, k), -1e9, dtype=np.float32)
        out_ids = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            cells = probe_cells[qi]
            vec_blocks = [self.invlists_vecs[c] for c in cells if self.invlists_vecs[c].shape[0] > 0]
            id_blocks = [self.invlists_ids[c] for c in cells if self.invlists_ids[c].shape[0] > 0]
            if not vec_blocks:
                continue
            cand_vecs = np.concatenate(vec_blocks, axis=0)
            cand_ids = np.concatenate(id_blocks, axis=0)
            sims = cand_vecs @ q[qi]
            kk = min(k, sims.shape[0])
            if sims.shape[0] > kk:
                part = np.argpartition(-sims, kth=kk - 1)[:kk]
                ord_k = part[np.argsort(-sims[part])]
            else:
                ord_k = np.argsort(-sims)
            out_scores[qi, :kk] = sims[ord_k]
            out_ids[qi, :kk] = cand_ids[ord_k]

        return out_scores, out_ids


class IVFPQIndex:
    """IVF with Residual Product Quantization (`IVF-PQ`) and Asymmetric Distance Computation (`ADC`).

    Each `D`-dimensional vector `x` assigned to coarse centroid `c` is encoded by quantizing its
    residual `r = x - c` across `M` orthogonal subspaces of dimension `d_sub = D // M`, each with
    `K* = 256` 8-bit codewords (`uint8`). At query time, squared L2 distance `||q - (c + r_hat)||^2`
    is evaluated via `(M, 256)` precomputed lookup tables per probed cell.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 256,
        m_subspaces: int = 32,
        k_centroids: int = 256,
        seed: int = 42,
    ) -> None:
        if dim % m_subspaces != 0:
            raise ValueError(f"dim={dim} must be divisible by m_subspaces={m_subspaces}")
        if k_centroids > 256:
            raise ValueError("uint8 PQ encoding supports k_centroids <= 256")
        self.dim = int(dim)
        self.nlist = int(nlist)
        self.m = int(m_subspaces)
        self.d_sub = self.dim // self.m
        self.k_sub = int(k_centroids)
        self.seed = int(seed)

        self.coarse_centroids: np.ndarray | None = None  # (nlist, D)
        self.pq_codebooks: np.ndarray | None = None      # (M, K*, d_sub)
        self.invlists_codes: list[np.ndarray] = []       # list of (n_c, M) uint8
        self.invlists_ids: list[np.ndarray] = []         # list of (n_c,) int64
        self._ntotal = 0

    @property
    def ntotal(self) -> int:
        return self._ntotal

    def memory_bytes(self) -> int:
        total = 0
        if self.coarse_centroids is not None:
            total += int(self.coarse_centroids.nbytes)
        if self.pq_codebooks is not None:
            total += int(self.pq_codebooks.nbytes)
        for c, i in zip(self.invlists_codes, self.invlists_ids):
            total += int(c.nbytes + i.nbytes)
        return total

    def code_bytes_only(self) -> int:
        return int(sum(c.nbytes for c in self.invlists_codes))

    def train_and_add(self, vectors: np.ndarray, train_sample_max: int = 20000) -> None:
        mat = _ensure_2d_f32(vectors)
        n, d = mat.shape
        eff_nlist = min(self.nlist, n)
        self.nlist = eff_nlist
        rng = np.random.default_rng(self.seed)

        train_idx = rng.choice(n, size=min(n, train_sample_max), replace=False)
        train_vecs = mat[train_idx]

        km_coarse = MiniBatchKMeans(
            n_clusters=eff_nlist,
            batch_size=2048,
            max_iter=50,
            n_init=1,
            random_state=self.seed,
        )
        km_coarse.fit(train_vecs)
        raw_c = km_coarse.cluster_centers_.astype(np.float32)
        c_norms = np.maximum(np.linalg.norm(raw_c, axis=1, keepdims=True), 1e-12)
        self.coarse_centroids = np.ascontiguousarray(raw_c / c_norms, dtype=np.float32)

        # Compute training residuals for PQ codebook learning
        tr_assign = np.argmax(train_vecs @ self.coarse_centroids.T, axis=1)
        tr_residuals = train_vecs - self.coarse_centroids[tr_assign]

        pq_cb = np.zeros((self.m, self.k_sub, self.d_sub), dtype=np.float32)
        pq_train_n = min(tr_residuals.shape[0], 12000)
        pq_sub_res = tr_residuals[:pq_train_n]
        for sub_m in range(self.m):
            s0 = sub_m * self.d_sub
            s1 = s0 + self.d_sub
            km_sub = MiniBatchKMeans(
                n_clusters=min(self.k_sub, pq_train_n),
                batch_size=2048,
                max_iter=35,
                n_init=1,
                random_state=self.seed + sub_m + 1,
            )
            km_sub.fit(pq_sub_res[:, s0:s1])
            pq_cb[sub_m, : km_sub.cluster_centers_.shape[0]] = km_sub.cluster_centers_.astype(np.float32)
        self.pq_codebooks = pq_cb

        # Encode full gallery in chunks
        all_assign = np.empty(n, dtype=np.int32)
        all_codes = np.empty((n, self.m), dtype=np.uint8)
        cb_norms_sq = np.sum(pq_cb * pq_cb, axis=2)  # (M, K*)

        chunk = 8192
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            sub_v = mat[s:e]
            c_idx = np.argmax(sub_v @ self.coarse_centroids.T, axis=1)
            all_assign[s:e] = c_idx
            res = sub_v - self.coarse_centroids[c_idx]  # (B, D)
            res_reshaped = res.reshape(-1, self.m, self.d_sub)  # (B, M, d_sub)
            # Squared L2 to each codebook entry: ||r_m||^2 + ||cb_m||^2 - 2 <r_m, cb_m>
            dots = np.einsum("bmd,mkd->bmk", res_reshaped, pq_cb)
            dists = cb_norms_sq[None, :, :] - 2.0 * dots
            all_codes[s:e] = np.argmin(dists, axis=2).astype(np.uint8)

        order = np.argsort(all_assign, kind="stable")
        sorted_assign = all_assign[order]
        counts = np.bincount(sorted_assign, minlength=eff_nlist)
        splits = np.cumsum(counts)[:-1]
        id_groups = np.split(order.astype(np.int64), splits)
        code_groups = np.split(all_codes[order], splits)

        self.invlists_ids = [np.ascontiguousarray(g) for g in id_groups]
        self.invlists_codes = [np.ascontiguousarray(cg) for cg in code_groups]
        self._ntotal = n

    def search(self, queries: np.ndarray, top_k: int = 10, nprobe: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self.coarse_centroids is None or self.pq_codebooks is None:
            raise RuntimeError("IVFPQIndex is not trained.")
        q = _ensure_2d_f32(queries)
        nq = q.shape[0]
        k = min(int(top_k), self._ntotal)
        eff_probe = min(max(1, int(nprobe)), self.nlist)

        coarse_sims = q @ self.coarse_centroids.T
        if eff_probe < self.nlist:
            probe_cells = np.argpartition(-coarse_sims, kth=eff_probe - 1, axis=1)[:, :eff_probe]
        else:
            probe_cells = np.tile(np.arange(self.nlist), (nq, 1))

        out_dists = np.full((nq, k), 1e9, dtype=np.float32)
        out_ids = np.full((nq, k), -1, dtype=np.int64)
        sub_arange = np.arange(self.m)

        # Precompute codebook squared norms (M, K*)
        cb = self.pq_codebooks
        cb_sq = np.sum(cb * cb, axis=2)  # (M, K*)

        for qi in range(nq):
            qv = q[qi]  # (D,)
            q_sub = qv.reshape(self.m, self.d_sub)  # (M, d_sub)
            # <q_m, cb_{m,k}> shared across all probed cells for query qi!
            q_cb_dots = np.einsum("md,mkd->mk", q_sub, cb)  # (M, K*)

            cand_dists_list: list[np.ndarray] = []
            cand_ids_list: list[np.ndarray] = []

            for cell_id in probe_cells[qi]:
                codes = self.invlists_codes[cell_id]
                if codes.shape[0] == 0:
                    continue
                # Residual query vector for cell: u = q - c_{cell}
                u_sub = (qv - self.coarse_centroids[cell_id]).reshape(self.m, self.d_sub)
                u_sq = np.sum(u_sub * u_sub, axis=1, keepdims=True)  # (M, 1)
                c_sub = self.coarse_centroids[cell_id].reshape(self.m, self.d_sub)
                c_cb_dots = np.einsum("md,mkd->mk", c_sub, cb)
                # Exact ADC table: ||(q - c) - cb_{m,k}||^2 = ||u_m||^2 + ||cb_{m,k}||^2 - 2 <q_m - c_m, cb_{m,k}>
                lut = u_sq + cb_sq - 2.0 * (q_cb_dots - c_cb_dots)  # (M, K*)
                approx_d = np.sum(lut[sub_arange, codes], axis=1)
                cand_dists_list.append(approx_d)
                cand_ids_list.append(self.invlists_ids[cell_id])

            if not cand_dists_list:
                continue
            all_d = np.concatenate(cand_dists_list, axis=0)
            all_i = np.concatenate(cand_ids_list, axis=0)
            kk = min(k, all_d.shape[0])
            if all_d.shape[0] > kk:
                part = np.argpartition(all_d, kth=kk - 1)[:kk]
                ord_k = part[np.argsort(all_d[part])]
            else:
                ord_k = np.argsort(all_d)
            out_dists[qi, :kk] = all_d[ord_k]
            out_ids[qi, :kk] = all_i[ord_k]

        # Return negated distances so higher score = closer match (consistent ranking convention)
        return -out_dists, out_ids


class HNSWIndex:
    """Hierarchical Navigable Small World (`HNSW`) graph index (unit-vector cosine / inner product)."""

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 100,
        seed: int = 42,
    ) -> None:
        self.dim = int(dim)
        self.m = int(m)
        self.m0 = int(2 * m)
        self.ef_construction = int(ef_construction)
        self.level_mult = 1.0 / math.log(max(2, self.m))
        self.rng = np.random.default_rng(seed)

        self._vectors: np.ndarray = np.empty((0, self.dim), dtype=np.float32)
        self._layers: list[dict[int, list[int]]] = []
        self._entry_point: int = -1
        self._max_level: int = -1
        self._visited_tag: np.ndarray = np.empty((0,), dtype=np.int32)
        self._visit_token: int = 1

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def memory_bytes(self) -> int:
        vec_b = int(self._vectors.nbytes)
        edge_count = sum(len(nbrs) for layer in self._layers for nbrs in layer.values())
        # 8 bytes per int64 neighbor ID + 4 bytes node header estimate
        graph_b = edge_count * 8 + sum(len(layer) * 16 for layer in self._layers)
        return vec_b + graph_b

    def _next_token(self) -> int:
        self._visit_token += 1
        if self._visit_token >= 2_000_000_000:
            self._visited_tag.fill(0)
            self._visit_token = 1
        return self._visit_token

    def _greedy_search_layer_1(self, q: np.ndarray, ep: int, level: int) -> int:
        curr = ep
        curr_dist = float(1.0 - np.dot(self._vectors[curr], q))
        layer_adj = self._layers[level]
        while True:
            nbrs = layer_adj.get(curr, [])
            if not nbrs:
                break
            nbr_arr = np.asarray(nbrs, dtype=np.int64)
            dists = 1.0 - (self._vectors[nbr_arr] @ q)
            best_pos = int(np.argmin(dists))
            if float(dists[best_pos]) < curr_dist:
                curr_dist = float(dists[best_pos])
                curr = int(nbr_arr[best_pos])
            else:
                break
        return curr

    def _search_layer_beam(self, q: np.ndarray, ep: int, ef: int, level: int) -> list[tuple[float, int]]:
        token = self._next_token()
        self._visited_tag[ep] = token
        d0 = float(1.0 - np.dot(self._vectors[ep], q))

        candidates: list[tuple[float, int]] = [(d0, ep)]  # min-heap on distance
        results: list[tuple[float, int]] = [(-d0, ep)]    # max-heap via negated distance
        layer_adj = self._layers[level]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_res_dist = -results[0][0]
            if c_dist > worst_res_dist and len(results) >= ef:
                break

            nbrs = layer_adj.get(c_id, [])
            if not nbrs:
                continue
            unvisited = [nb for nb in nbrs if self._visited_tag[nb] != token]
            if not unvisited:
                continue
            for nb in unvisited:
                self._visited_tag[nb] = token

            nb_arr = np.asarray(unvisited, dtype=np.int64)
            dists = 1.0 - (self._vectors[nb_arr] @ q)

            for d_val, nb_id in zip(dists.tolist(), unvisited):
                if len(results) < ef or d_val < -results[0][0]:
                    heapq.heappush(candidates, (d_val, nb_id))
                    heapq.heappush(results, (-d_val, nb_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        out = [(-neg_d, nid) for neg_d, nid in results]
        out.sort(key=lambda x: x[0])
        return out

    def build(self, vectors: np.ndarray) -> None:
        mat = _ensure_2d_f32(vectors)
        n, d = mat.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        self._vectors = mat.copy()
        self._visited_tag = np.zeros(n, dtype=np.int32)
        self._visit_token = 1
        self._layers = []
        self._entry_point = -1
        self._max_level = -1

        u = np.clip(self.rng.random(n), 1e-12, 1.0)
        node_levels = np.floor(-np.log(u) * self.level_mult).astype(np.int32)

        for idx in range(n):
            lvl = int(node_levels[idx])
            while len(self._layers) <= lvl:
                self._layers.append({})
            for l_i in range(lvl + 1):
                self._layers[l_i][idx] = []

            if self._entry_point < 0:
                self._entry_point = idx
                self._max_level = lvl
                continue

            qv = self._vectors[idx]
            curr_ep = self._entry_point

            for l_c in range(self._max_level, lvl, -1):
                curr_ep = self._greedy_search_layer_1(qv, curr_ep, l_c)

            for l_c in range(min(lvl, self._max_level), -1, -1):
                m_max = self.m0 if l_c == 0 else self.m
                found = self._search_layer_beam(qv, curr_ep, self.ef_construction, l_c)
                chosen = [nid for _, nid in found[: self.m] if nid != idx]
                self._layers[l_c][idx] = chosen
                if chosen:
                    curr_ep = chosen[0]

                layer_adj = self._layers[l_c]
                for nb in chosen:
                    nb_list = layer_adj[nb]
                    nb_list.append(idx)
                    if len(nb_list) > m_max:
                        nb_arr = np.asarray(nb_list, dtype=np.int64)
                        d_nb = 1.0 - (self._vectors[nb_arr] @ self._vectors[nb])
                        keep_order = np.argsort(d_nb)[:m_max]
                        layer_adj[nb] = [int(nb_arr[p]) for p in keep_order]

            if lvl > self._max_level:
                self._max_level = lvl
                self._entry_point = idx

    def search(self, queries: np.ndarray, top_k: int = 10, ef_search: int = 64) -> tuple[np.ndarray, np.ndarray]:
        q = _ensure_2d_f32(queries)
        nq = q.shape[0]
        k = min(int(top_k), self.ntotal)
        eff_ef = max(k, int(ef_search))

        out_scores = np.full((nq, k), -1e9, dtype=np.float32)
        out_ids = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            qv = q[qi]
            curr_ep = self._entry_point
            for l_c in range(self._max_level, 0, -1):
                curr_ep = self._greedy_search_layer_1(qv, curr_ep, l_c)
            found = self._search_layer_beam(qv, curr_ep, eff_ef, 0)
            kk = min(k, len(found))
            for j in range(kk):
                dist_val, nid = found[j]
                out_scores[qi, j] = 1.0 - dist_val
                out_ids[qi, j] = nid

        return out_scores, out_ids


def compute_recall_at_k(
    ground_truth_top1_or_topk: np.ndarray,
    predicted_ids: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Compute Recall@K (proportion of queries whose true exact top-1 neighbor appears in top-K)."""
    gt_top1 = ground_truth_top1_or_topk[:, 0] if ground_truth_top1_or_topk.ndim == 2 else ground_truth_top1_or_topk
    nq = int(gt_top1.shape[0])
    metrics: dict[str, Any] = {}
    ranks: list[int] = []

    for qi in range(nq):
        matches = np.where(predicted_ids[qi] == gt_top1[qi])[0]
        ranks.append(int(matches[0] + 1) if matches.size > 0 else -1)

    ranks_arr = np.asarray(ranks, dtype=np.int32)
    for kv in k_values:
        eff_k = min(kv, predicted_ids.shape[1])
        hits = np.any(predicted_ids[:, :eff_k] == gt_top1[:, None], axis=1)
        metrics[f"recall_at_{kv}"] = round(float(np.mean(hits)), 6)

    found_mask = ranks_arr > 0
    metrics["rank_distribution"] = {
        "found_within_returned_pct": round(float(np.mean(found_mask) * 100.0), 4),
        "rank_1_count": int(np.sum(ranks_arr == 1)),
        "rank_2_to_5_count": int(np.sum((ranks_arr >= 2) & (ranks_arr <= 5))),
        "rank_6_to_10_count": int(np.sum((ranks_arr >= 6) & (ranks_arr <= 10))),
        "missed_count": int(np.sum(~found_mask)),
    }
    return metrics
