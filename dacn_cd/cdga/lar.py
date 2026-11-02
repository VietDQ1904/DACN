"""Biểu diễn LAR và các toán tử di truyền (Chương 2, mục 2.2 và 2.3.6)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .graph import Graph


class LARFactory:
    """Khởi tạo / lai ghép / đột biến / mã hóa ngược cho nhiễm sắc thể LAR."""

    def __init__(self, g: Graph, rho: float):
        self.g = g
        self.rho = rho
        cc = g.clustering_coefficient()
        # N_high(i): láng giềng có CC >= CC trung bình của N(i) (mục 2.2.2).
        hi_idx, hi_cnt = [], np.zeros(g.n, dtype=np.int64)
        for i in range(g.n):
            nb = g.neighbors(i)
            c = cc[nb]
            sel = nb[c >= c.mean() - 1e-12]
            hi_idx.append(sel)
            hi_cnt[i] = len(sel)
        self.hi_cnt = hi_cnt
        self.hi_ptr = np.concatenate([[0], np.cumsum(hi_cnt)])[:-1]
        self.hi_idx = np.concatenate(hi_idx).astype(np.int32)

    # ---- khởi tạo (mục 2.2.2) ---------------------------------------------
    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        g = self.g
        use_high = rng.random(g.n) < self.rho
        r1, r2 = rng.random(g.n), rng.random(g.n)
        hi = self.hi_idx[self.hi_ptr + (r1 * self.hi_cnt).astype(np.int64)]
        lo = g.indices[g.indptr[:-1] + (r2 * g.deg).astype(np.int64)]
        return np.where(use_high, hi, lo).astype(np.int32)

    def init_population(self, mu: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([self.random_individual(rng) for _ in range(mu)])

    # ---- lai ghép đồng nhất (mục 2.2.5) ----------------------------------
    @staticmethod
    def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(p1.shape) < 0.5
        return np.where(mask, p1, p2)

    # ---- đột biến láng giềng (mục 2.2.6) ----------------------------------
    def mutate(self, genes: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
        """Mỗi gen đột biến với xác suất pm sang một láng giềng KHÁC g_i (nếu k_i >= 2)."""
        g = self.g
        flip = (rng.random(g.n) < pm) & (g.deg > 1)
        if not flip.any():
            return genes
        out = genes.copy()
        i = np.flatnonzero(flip)
        pos = np.searchsorted(g.keys, i.astype(np.int64) * g.n + genes[i].astype(np.int64)) - g.indptr[i]
        k = g.deg[i]
        new_pos = (pos + 1 + (rng.random(len(i)) * (k - 1)).astype(np.int64)) % k
        out[i] = g.indices[g.indptr[i] + new_pos]
        return out

    # ---- mã hóa ngược (Thuật toán 3) --------------------------------------
    def reencode(self, labels: np.ndarray) -> np.ndarray:
        """Cây khung BFS trên từng cộng đồng. Yêu cầu: mỗi cộng đồng liên thông và có >= 2 đỉnh.

        g_u = cha của u trong cây BFS; đỉnh gốc r (bậc lớn nhất trong cộng đồng)
        nhận một láng giềng cùng cộng đồng (con đầu tiên của nó).
        """
        g = self.g
        adj = g.adj_lists
        genes = np.full(g.n, -1, dtype=np.int32)
        K = int(labels.max()) + 1
        members = [[] for _ in range(K)]
        for u, c in enumerate(labels.tolist()):
            members[c].append(u)
        lab = labels.tolist()
        for c in range(K):
            mem = members[c]
            if len(mem) < 2:
                raise ValueError("Reencode: cộng đồng có < 2 đỉnh (singleton không biểu diễn được bằng LAR)")
            root = max(mem, key=lambda x: g.deg[x])
            seen = {root}
            dq = deque([root])
            first_child = -1
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if lab[v] == c and v not in seen:
                        seen.add(v)
                        genes[v] = u
                        if first_child < 0:
                            first_child = v
                        dq.append(v)
            if len(seen) != len(mem):
                raise ValueError("Reencode: cộng đồng không liên thông")
            genes[root] = first_child
        return genes
