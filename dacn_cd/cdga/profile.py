"""Hồ sơ thống kê của một bộ dữ liệu (đã tiền xử lý) để vẽ biểu đồ mô tả dữ liệu và điền Bảng 2.3.

`dataset_profile(g)` trả về dict JSON-được; `save_graph_npz` lưu cạnh + nhãn chuẩn để vẽ đồ thị phân hoạch
mà không cần tệp dữ liệu gốc ở bước phân tích.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .graph import Graph


def safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def dataset_profile(g: Graph) -> dict:
    deg = g.deg
    A = g.adjacency()
    tri = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2.0     # số tam giác chứa từng đỉnh
    triplets = float((deg * (deg - 1) / 2.0).sum())
    cc = g.clustering_coefficient()
    d_vals, d_cnt = np.unique(deg, return_counts=True)
    prof = {
        "name": g.name, "n": int(g.n), "m": int(g.m),
        "avg_degree": float(2.0 * g.m / g.n), "median_degree": float(np.median(deg)),
        "min_degree": int(deg.min()), "max_degree": int(deg.max()),
        "density": float(2.0 * g.m / (g.n * (g.n - 1.0))) if g.n > 1 else 0.0,
        "avg_cc": float(cc.mean()),
        "transitivity": float(tri.sum() / triplets) if triplets > 0 else 0.0,
        "degree_values": d_vals.tolist(), "degree_counts": d_cnt.tolist(),
        "cc_hist": np.histogram(cc, bins=20, range=(0.0, 1.0))[0].tolist(),
        "raw": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else (int(v) if isinstance(v, (int, np.integer)) else v))
                for k, v in g.stats.items() if k in ("n_raw", "m_raw_input", "m_raw_undirected", "directed_raw",
                                                      "n_truth_communities", "n_truth_labeled")},
        "truth_K": None, "truth_sizes": [], "n_labeled": 0, "truth_Q": None, "truth_Phi": None, "mixing": None,
    }
    t = g.truth
    if t is not None and (t >= 0).any():
        lab = t >= 0
        sizes = np.bincount(t[lab])
        prof["truth_K"] = int((sizes > 0).sum())
        prof["truth_sizes"] = sorted(sizes[sizes > 0].astype(int).tolist(), reverse=True)
        prof["n_labeled"] = int(lab.sum())
        both = lab[g.src] & lab[g.dst]                                       # cạnh có cả hai đầu có nhãn
        if both.any():
            prof["mixing"] = float((t[g.src][both] != t[g.dst][both]).mean())   # μ thực nghiệm: tỉ lệ cạnh liên cộng đồng
        if lab.all() and t.max() >= 1:
            q, phi = g.objectives(t.astype(np.int32))
            prof["truth_Q"], prof["truth_Phi"] = float(q), float(phi)
    return prof


def save_graph_npz(g: Graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    truth = g.truth if g.truth is not None else np.full(g.n, -1, dtype=np.int32)
    np.savez_compressed(path, src=g.src, dst=g.dst, truth=truth, n=np.int64(g.n))
