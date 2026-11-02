"""Sinh dữ liệu MÔ PHỎNG đúng định dạng tệp của Newman/SNAP để kiểm thử bộ nạp (KHÔNG phải dữ liệu thật)."""
from __future__ import annotations

import gzip
from pathlib import Path

import networkx as nx
import numpy as np


def _planted(sizes, p_in, p_out, seed):
    G = nx.stochastic_block_model(sizes, [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))], seed=seed)
    G.graph.clear()   # bỏ thuộc tính 'partition' (tập hợp) để ghi được GML
    return G, [b for b, s in enumerate(sizes) for _ in range(s)]


def build(d: Path, email_n=300, amazon_n=1500):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # football.gml: vô hướng, thuộc tính label (tên đội) và value (liên đoàn)
    G, comm = _planted([10] * 6, 0.75, 0.04, 1)
    for v in G:
        G.nodes[v].clear()
        G.nodes[v].update(label=f"Team{v}", value=comm[v])
    nx.write_gml(nx.convert_node_labels_to_integers(G), d / "football.gml")

    # polbooks.gml: nhãn chuỗi l / n / c
    G, comm = _planted([40, 20, 40], 0.18, 0.015, 2)
    for v in G:
        G.nodes[v].clear()
        G.nodes[v].update(label=f"Book {v}", value="lnc"[comm[v]])
    nx.write_gml(G, d / "polbooks.gml")

    # polblogs.gml: CÓ HƯỚNG, có self-loop, cạnh hai chiều, đỉnh cô lập và thành phần phụ
    G, comm = _planted([60, 60], 0.12, 0.01, 3)
    D = nx.DiGraph()
    for u, v in G.edges():
        D.add_edge(u, v)
        if rng.random() < 0.4:
            D.add_edge(v, u)
    D.add_edge(0, 0)
    D.add_node(500)               # cô lập
    D.add_edge(900, 901)          # thành phần phụ
    for v in D:
        D.nodes[v].update(value=comm[v] if v < 120 else 0, source="blog")
    nx.write_gml(D, d / "polblogs.gml")

    # email-Eu-core: cạnh có hướng, self-loop; nhãn phòng ban (mọi đỉnh)
    G, comm = _planted([email_n // 6] * 6, 0.25, 0.015, 4)
    with open(d / "email-Eu-core.txt", "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
            if rng.random() < 0.5:
                f.write(f"{v} {u}\n")
        f.write("0 0\n")
    with open(d / "email-Eu-core-department-labels.txt", "w") as f:
        for v in sorted(G):
            f.write(f"{v} {comm[v]}\n")

    # com-amazon: chú thích '#', cộng đồng CHỒNG LẤP, nhiều đỉnh không thuộc cộng đồng nào
    sizes = [amazon_n // 30] * 30
    G, comm = _planted(sizes, 0.12, 0.0004, 5)
    with gzip.open(d / "com-amazon.ungraph.txt.gz", "wt") as f:
        f.write("# Undirected graph (mô phỏng)\n# FromNodeId\tToNodeId\n")
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")
    members = [[v for v in G if comm[v] == c] for c in range(30)]
    members[0] += members[1][:5]          # chồng lấp: 5 đỉnh thuộc 2 cộng đồng
    members = [m for i, m in enumerate(members) if i % 5 != 4]   # bỏ một số cộng đồng => đỉnh không nhãn
    with open(d / "com-amazon.all.dedup.cmty.txt", "w") as f:
        for m in members:
            f.write("\t".join(map(str, m)) + "\n")
    return d
