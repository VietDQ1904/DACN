"""Tối ưu đa mục tiêu: trội Pareto, NSGA-II, khoảng cách đám đông, Hypervolume.

Quy ước: mọi vectơ mục tiêu F đều được ĐƯA VỀ CÙNG HƯỚNG CỰC TIỂU
(F = (1 - Q, Φ), xem mục 2.2.8). Hypervolume dùng điểm tham chiếu cố định.
"""
from __future__ import annotations

import numpy as np

HV_REF = np.array([1.1, 1.1])   # điểm tham chiếu cố định trong không gian (1-Q, Φ)


def to_min(q, phi, multi: bool = True):
    """(Q, Φ) -> F cực tiểu. Chế độ đơn mục tiêu: chỉ còn Q (cột thứ hai = 0)."""
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.stack([1.0 - q, phi if multi else np.zeros_like(phi)], axis=-1)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def domination_matrix(F: np.ndarray) -> np.ndarray:
    """D[i, j] = True nếu i trội j."""
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return le & lt


def nd_indices(F: np.ndarray) -> np.ndarray:
    """Chỉ số các điểm không bị trội (giữ cả các điểm trùng giá trị)."""
    if len(F) == 0:
        return np.array([], dtype=int)
    D = domination_matrix(F)
    return np.flatnonzero(~D.any(axis=0))


def nd_sort(F: np.ndarray) -> np.ndarray:
    """Pareto rank (0 = front tốt nhất), sắp xếp không trội kiểu NSGA-II, O(N^2)."""
    N = len(F)
    D = domination_matrix(F)
    dom_count = D.sum(axis=0).astype(int)          # số điểm trội i
    rank = np.full(N, -1, dtype=int)
    current = np.flatnonzero(dom_count == 0)
    r = 0
    while len(current):
        rank[current] = r
        nxt = []
        for i in current:
            for j in np.flatnonzero(D[i]):
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = np.array(nxt, dtype=int)
        r += 1
    return rank


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Khoảng cách đám đông của một tập điểm (thường là một front)."""
    N, M = F.shape
    dist = np.zeros(N)
    if N <= 2:
        dist[:] = np.inf
        return dist
    for k in range(M):
        order = np.argsort(F[:, k], kind="stable")
        fk = F[order, k]
        rng = fk[-1] - fk[0]
        dist[order[0]] = dist[order[-1]] = np.inf
        if rng <= 0:
            continue
        dist[order[1:-1]] += (fk[2:] - fk[:-2]) / rng
    return dist


def rank_and_crowding(F: np.ndarray):
    rank = nd_sort(F)
    crowd = np.zeros(len(F))
    for r in range(rank.max() + 1):
        idx = np.flatnonzero(rank == r)
        crowd[idx] = crowding_distance(F[idx])
    return rank, crowd


def nsga_select(F: np.ndarray, mu: int):
    """Chọn mu cá thể: theo rank rồi theo crowding. Trả về (chỉ số, rank, crowd) của mu cá thể chọn."""
    rank, crowd = rank_and_crowding(F)
    order = np.lexsort((-crowd, rank))   # rank tăng, crowd giảm
    sel = order[:mu]
    return sel, rank[sel], crowd[sel]


def hypervolume(F: np.ndarray, ref: np.ndarray = HV_REF) -> float:
    """Hypervolume 2 mục tiêu (cực tiểu) so với điểm tham chiếu ref."""
    if len(F) == 0:
        return 0.0
    P = F[np.all(F < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    hv, best_f2 = 0.0, ref[1]
    for f1, f2 in P:
        if f2 < best_f2:
            hv += (ref[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return float(hv)


def truncate_by_crowding(F: np.ndarray, size: int) -> np.ndarray:
    """Giữ `size` điểm bằng cách lần lượt loại điểm có khoảng cách đám đông nhỏ nhất."""
    keep = np.arange(len(F))
    while len(keep) > size:
        cd = crowding_distance(F[keep])
        finite = np.where(np.isinf(cd), np.inf, cd)
        keep = np.delete(keep, int(np.argmin(finite)))
    return keep


def knee_index(F: np.ndarray) -> int:
    """Điểm gần điểm lý tưởng nhất sau chuẩn hóa min-max (quy tắc chọn đại diện)."""
    if len(F) == 1:
        return 0
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    Z = (F - lo) / span
    return int(np.argmin(np.sqrt((Z ** 2).sum(axis=1))))
