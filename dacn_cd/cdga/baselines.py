"""Baseline heuristic Louvain và Leiden (mục 2.5.3).

Louvain: networkx.community.louvain_communities (không cần cài thêm).
Leiden : cần `leidenalg` + `igraph` (pip install igraph leidenalg). KHÔNG dùng
         networkx.community.leiden_communities làm phương án dự phòng: trong thử nghiệm với
         networkx 3.7 nó trả về toàn bộ đỉnh đơn lẻ (Q ≈ 0), không đáng tin cậy.
Mọi kết quả được đánh số lại thành nhãn 0..K-1 trên đồ thị đã tiền xử lý.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .graph import Graph


def to_networkx(g: Graph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(zip(g.src.tolist(), g.dst.tolist()))
    return G


def _labels_from_sets(g: Graph, comms) -> np.ndarray:
    lab = np.full(g.n, -1, dtype=np.int32)
    for c, nodes in enumerate(comms):
        lab[list(nodes)] = c
    assert (lab >= 0).all()
    return lab


def louvain(g: Graph, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    comms = nx.community.louvain_communities(to_networkx(g), seed=seed, resolution=resolution)
    return _labels_from_sets(g, comms)


def leiden_available() -> bool:
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        return True
    except ImportError:
        return False


def leiden(g: Graph, seed: int = 0) -> tuple[np.ndarray, str]:
    """Trả về (nhãn, tên cài đặt đã dùng). Tối ưu Modularity (ModularityVertexPartition)."""
    if not leiden_available():
        raise RuntimeError("Thiếu thư viện: chạy `pip install igraph leidenalg` để dùng baseline Leiden.")
    import igraph as ig
    import leidenalg
    G = ig.Graph(n=g.n, edges=list(zip(g.src.tolist(), g.dst.tolist())))
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, seed=seed)
    return np.array(part.membership, dtype=np.int32), "leidenalg"
