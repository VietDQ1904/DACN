"""Đồ thị đã tiền xử lý (CSR) và các hàm mục tiêu Q, Φ (Chương 2, mục 2.1.3).

Tiền xử lý theo mục 2.1.2: đồ thị vô hướng, không trọng số, bỏ self-loop và
cạnh song song, chỉ giữ thành phần liên thông lớn nhất (LAR không biểu diễn được
đỉnh cô lập), sau đó đánh số lại đỉnh 0..n-1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Giá trị phạt cho phân hoạch không hợp lệ (K < 2): bị mọi nghiệm hợp lệ trội.
INVALID_Q = -0.5
INVALID_PHI = 1.0


@dataclass
class Graph:
    n: int
    m: int
    indptr: np.ndarray            # (n+1,) CSR
    indices: np.ndarray           # (2m,) láng giềng, đã sắp xếp theo từng hàng
    deg: np.ndarray               # (n,)
    src: np.ndarray               # (m,) mỗi cạnh vô hướng đúng một lần (src < dst)
    dst: np.ndarray               # (m,)
    name: str = "graph"
    truth: np.ndarray | None = None   # nhãn ground-truth (đã đánh số lại); -1 = đỉnh không có nhãn
    orig_labels: list = field(default_factory=list)  # nhãn đỉnh gốc theo thứ tự mới
    stats: dict = field(default_factory=dict)         # thống kê trước/sau tiền xử lý

    # ---- tiện ích -------------------------------------------------------
    def neighbors(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    @property
    def adj_lists(self) -> list[list[int]]:
        if not hasattr(self, "_adj_lists"):
            self._adj_lists = [self.neighbors(u).tolist() for u in range(self.n)]
        return self._adj_lists

    @property
    def keys(self) -> np.ndarray:
        """Khóa i*n + j đã sắp xếp, dùng để tìm vị trí của láng giềng trong hàng."""
        if not hasattr(self, "_keys"):
            rows = np.repeat(np.arange(self.n, dtype=np.int64), self.deg)
            self._keys = rows * self.n + self.indices.astype(np.int64)
        return self._keys

    # ---- hệ số phân cụm (mục 2.2.2) --------------------------------------
    def clustering_coefficient(self) -> np.ndarray:
        """CC(i) = 2 T_i / (k_i (k_i - 1)); CC = 0 nếu k_i < 2."""
        if not hasattr(self, "_cc"):
            A = self.adjacency()
            tri = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2.0  # T_i
            k = self.deg.astype(float)
            denom = k * (k - 1.0)
            cc = np.zeros(self.n)
            ok = denom > 0
            cc[ok] = 2.0 * tri[ok] / denom[ok]
            self._cc = cc
        return self._cc

    def adjacency(self) -> sp.csr_matrix:
        if not hasattr(self, "_A"):
            data = np.ones(len(self.indices), dtype=np.float64)
            self._A = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))
        return self._A

    # ---- giải mã LAR ------------------------------------------------------
    def decode(self, genes: np.ndarray) -> np.ndarray:
        """Thành phần liên thông yếu của G_g = (V, {(i, g_i)}); O(n)."""
        A = sp.coo_matrix((np.ones(self.n, dtype=np.int8), (np.arange(self.n), genes)),
                          shape=(self.n, self.n))
        _, labels = connected_components(A, directed=True, connection="weak")
        return labels.astype(np.int32)

    # ---- hàm mục tiêu -----------------------------------------------------
    def community_stats(self, labels: np.ndarray, K: int | None = None):
        """Trả về (l_c, d_c): số cạnh nội bộ và tổng bậc của từng cộng đồng."""
        if K is None:
            K = int(labels.max()) + 1
        same = labels[self.src] == labels[self.dst]
        l = np.bincount(labels[self.src][same], minlength=K).astype(np.float64)
        d = np.bincount(labels, weights=self.deg, minlength=K)
        return l, d

    def objectives(self, labels: np.ndarray) -> tuple[float, float]:
        """(Q, Φ). `labels` phải được đánh số liên tục 0..K-1.

        Phân hoạch một cộng đồng (K < 2) không hợp lệ -> trả về giá trị phạt.
        """
        K = int(labels.max()) + 1
        if K < 2:
            return INVALID_Q, INVALID_PHI
        l, d = self.community_stats(labels, K)
        return self.q_phi_from_stats(l, d)

    def q_phi_from_stats(self, l: np.ndarray, d: np.ndarray) -> tuple[float, float]:
        m2 = 2.0 * self.m
        q = float(np.sum(l / self.m - (d / m2) ** 2))
        cut = d - 2.0 * l
        denom = np.minimum(d, m2 - d)
        phi = float(np.mean(cut / denom))
        return q, phi

    def conductance_of(self, l: float, d: float) -> float:
        """φ(C_c) từ (l_c, d_c)."""
        m2 = 2.0 * self.m
        return (d - 2.0 * l) / min(d, m2 - d)


def canonical(labels: np.ndarray) -> np.ndarray:
    """Đánh số lại nhãn theo thứ tự xuất hiện đầu tiên (0..K-1)."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv].astype(np.int32)


def partition_key(labels: np.ndarray) -> bytes:
    return canonical(labels).tobytes()


def from_networkx(G: nx.Graph, name: str = "graph", truth: dict | None = None,
                  lcc: bool = True) -> Graph:
    """Tiền xử lý đồ thị networkx -> Graph (CSR)."""
    stats = {"n_raw": G.number_of_nodes(), "m_raw_input": G.number_of_edges(),
             "directed_raw": bool(G.is_directed())}
    G = nx.Graph(G)  # vô hướng, gộp cạnh song song và hai chiều
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    stats["m_raw_undirected"] = G.number_of_edges()
    G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
    if lcc and G.number_of_nodes() > 0:
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()
    nodes = sorted(G.nodes(), key=lambda x: (str(type(x)), x))
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
    a = np.minimum(edges[:, 0], edges[:, 1])
    b = np.maximum(edges[:, 0], edges[:, 1])
    order = np.lexsort((b, a))
    src, dst = a[order], b[order]
    m = len(src)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    o = np.lexsort((cols, rows))
    rows, cols = rows[o], cols[o]
    deg = np.bincount(rows, minlength=n).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
    t = None
    n_labeled = 0
    if truth is not None:
        raw = [truth.get(v) for v in nodes]
        names = sorted({str(x) for x in raw if x is not None})
        if names:
            code = {nm: i for i, nm in enumerate(names)}
            t = np.array([code[str(x)] if x is not None else -1 for x in raw], dtype=np.int32)
            n_labeled = int((t >= 0).sum())
    stats.update(n=n, m=m, avg_degree=2.0 * m / n,
                 avg_cc=float(np.nan))
    g = Graph(n=n, m=m, indptr=indptr, indices=cols.astype(np.int32), deg=deg,
              src=src.astype(np.int32), dst=dst.astype(np.int32), name=name,
              truth=t, orig_labels=nodes, stats=stats)
    g.stats["avg_cc"] = float(g.clustering_coefficient().mean())
    g.stats["n_truth_communities"] = int(t.max() + 1) if t is not None else None
    g.stats["n_truth_labeled"] = n_labeled
    return g
