"""Nạp bộ dữ liệu (mục 2.5.1) và sinh dữ liệu tổng hợp LFR.

Thư mục dữ liệu: `data/` cạnh `cdga/` (đổi bằng biến môi trường DACN_DATA). Tên tệp không phân
biệt hoa/thường và có thể nén `.gz`.

  karate    networkx.karate_club_graph() (có nhãn 2 nhóm). `karate.gml` của Newman không có nhãn.
  football  football.gml                           (Newman; nhãn = thuộc tính `value`, 12 liên đoàn)
  polbooks  polbooks.gml                           (Newman; nhãn `value` l/n/c)
  polblogs  polblogs.gml                           (Newman; có hướng -> vô hướng hóa; nhãn `value` 0/1)
  email     email-Eu-core.txt                      (SNAP; danh sách cạnh có hướng)
            email-Eu-core-department-labels.txt    (SNAP; "đỉnh phòng_ban")
  amazon    com-amazon.ungraph.txt                 (SNAP; danh sách cạnh)
            amazon:top=20 -> đồ thị con từ 20 cộng đồng chuẩn lớn nhất (đủ nhỏ để chạy GA)
            com-amazon.all.dedup.cmty.txt          (SNAP; mỗi dòng một cộng đồng, CHỒNG LẤP)
            (hoặc com-amazon.top5000.cmty.txt)
  lfr:n=250,mu=0.3,seed=1   đồ thị LFR sinh bằng networkx (ground-truth = 'community')

Nhãn chồng lấp (Amazon): NMI/ARI chỉ tính trên các đỉnh thuộc ĐÚNG MỘT cộng đồng chuẩn; các đỉnh
thuộc 0 hoặc >= 2 cộng đồng nhận nhãn -1 và bị loại khỏi NMI/ARI (xem metrics.py).
"""
from __future__ import annotations

import gzip
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np

from .graph import Graph, from_networkx

SNAP_HINT = {
    "email": "https://snap.stanford.edu/data/email-Eu-core.html  (email-Eu-core.txt.gz, email-Eu-core-department-labels.txt.gz)",
    "amazon": "https://snap.stanford.edu/data/com-Amazon.html  (com-amazon.ungraph.txt.gz, com-amazon.all.dedup.cmty.txt.gz)",
    "newman": "https://public.websites.umich.edu/~mejn/netdata  (polbooks.zip, football.zip, polblogs.zip -> lấy tệp .gml)",
}


def data_dir() -> Path:
    return Path(os.environ.get("DACN_DATA", Path(__file__).resolve().parents[1] / "data"))


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _find(names, hint: str) -> Path:
    d = data_dir()
    files = {p.name.lower(): p for p in d.iterdir()} if d.exists() else {}
    for n in names:
        for cand in (n, n + ".gz"):
            if cand.lower() in files:
                return files[cand.lower()]
    raise FileNotFoundError(f"Không thấy {' hoặc '.join(names)} trong {d}.\n  Tải tại: {hint}")


def _edge_graph(path: Path) -> nx.Graph:
    G = nx.Graph()
    with _open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            u, v = line.split()[:2]
            G.add_edge(int(u), int(v))
    return G


# ----------------------------------------------------------------------
def load_karate() -> Graph:
    G = nx.karate_club_graph()
    return from_networkx(G, "karate", {v: d["club"] for v, d in G.nodes(data=True)})


def _read_gml(path: Path) -> nx.Graph:
    """nx.read_gml từ chối tệp không phải ASCII (tên sách/blog có dấu nháy cong, chữ có dấu...).

    Đọc thô, giải mã utf-8 (hoặc latin-1) rồi đổi ký tự ngoài ASCII sang &#NNN; trước khi phân tích.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.encode("ascii", "xmlcharrefreplace").decode("ascii")
    try:
        return nx.parse_gml(text, label="id")
    except nx.NetworkXError as exc:
        if "is duplicated" not in str(exc):
            raise
        multigraph_text, count = re.subn(
            r"(?m)^(\s*graph\s*\[)",
            r"\1\n  multigraph 1",
            text,
            count=1,
        )
        if not count:
            raise
        return nx.parse_gml(multigraph_text, label="id")


def load_gml(names, name: str) -> Graph:
    G = _read_gml(_find(names, SNAP_HINT["newman"]))
    truth = {v: d.get("value") for v, d in G.nodes(data=True)}
    return from_networkx(G, name, truth if any(x is not None for x in truth.values()) else None)


def load_email() -> Graph:
    G = _edge_graph(_find(["email-Eu-core.txt"], SNAP_HINT["email"]))   # (u,v) và (v,u) gộp thành 1 cạnh
    truth = {}
    with _open(_find(["email-Eu-core-department-labels.txt"], SNAP_HINT["email"])) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                u, c = line.split()[:2]
                truth[int(u)] = int(c)
    return from_networkx(G, "email", truth)


def _read_communities(path: Path) -> list[list[int]]:
    out = []
    with _open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                out.append([int(t) for t in line.split()])
    return out


def load_amazon(with_truth: bool = True, top: int | None = None) -> Graph:
    """com-Amazon (SNAP).

    top = None : toàn bộ đồ thị (334.863 đỉnh) - chỉ Louvain/Leiden chạy nổi.
    top = K    : đồ thị con cảm sinh bởi hợp của K cộng đồng chuẩn LỚN NHẤT (để chạy cả các thuật toán GA).
    Ground-truth chồng lấp: chỉ đỉnh thuộc đúng một cộng đồng (được xét) mới có nhãn, còn lại -1.
    """
    G = _edge_graph(_find(["com-amazon.ungraph.txt"], SNAP_HINT["amazon"]))
    comms = None
    if with_truth or top:
        try:
            cm = _find(["com-amazon.all.dedup.cmty.txt", "com-amazon.top5000.cmty.txt"], SNAP_HINT["amazon"])
            comms = _read_communities(cm)
        except FileNotFoundError:
            if top:
                raise
    name = "amazon"
    if top:
        comms = sorted(comms, key=len, reverse=True)[:top]
        G = G.subgraph(set().union(*comms)).copy()
        name = f"amazon_top{top}"
    truth = None
    if comms:
        count: dict[int, int] = {}
        label: dict[int, int] = {}
        for cid, members in enumerate(comms):
            for v in members:
                count[v] = count.get(v, 0) + 1
                label[v] = cid
        truth = {v: label[v] for v, c in count.items() if c == 1}   # chỉ đỉnh thuộc đúng 1 cộng đồng
    return from_networkx(G, name, truth)


def make_lfr(n=250, mu=0.3, seed=1, tau1=3.0, tau2=1.5, avg_deg=10, max_deg=40,
             min_comm=20, max_comm=60) -> Graph:
    G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg,
                               min_community=min_comm, max_community=max_comm, seed=seed)
    truth = {v: min(d["community"]) for v, d in G.nodes(data=True)}     # frozenset -> id ổn định
    return from_networkx(G, f"lfr_mu{mu}_s{seed}", truth)


def load(name: str) -> Graph:
    key = name.lower()
    if key == "karate":
        return load_karate()
    if key in ("polbooks", "football", "polblogs"):
        return load_gml([f"{key}.gml"], key)
    if key == "email":
        return load_email()
    if key.startswith("amazon"):
        top = None
        if ":" in key:                      # amazon:top=20
            k, v = key.split(":", 1)[1].split("=")
            if k != "top":
                raise ValueError("amazon chỉ nhận tham số top=K, ví dụ amazon:top=20")
            top = int(v)
        return load_amazon(top=top)
    if key.startswith("lfr"):
        kw = {}
        if ":" in name:
            for part in name.split(":", 1)[1].split(","):
                k, v = part.split("=")
                kw[k] = float(v) if k in ("mu", "tau1", "tau2") else int(v)
        return make_lfr(**kw)
    raise ValueError(f"Bộ dữ liệu không rõ: {name} (karate, polbooks, football, polblogs, email, amazon, lfr:...)")
