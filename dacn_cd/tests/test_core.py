"""Kiểm thử tính đúng của các thành phần lõi. Chạy: python -m pytest tests -q  (hoặc python tests/test_core.py)"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DACN.dacn_cd.cdga import pareto                                   # noqa: E402
from DACN.dacn_cd.cdga.datasets import load_karate, make_lfr           # noqa: E402
from DACN.dacn_cd.cdga.ga import Config, Engine, select_solutions      # noqa: E402
from DACN.dacn_cd.cdga.graph import canonical                          # noqa: E402
from DACN.dacn_cd.cdga.lar import LARFactory                           # noqa: E402
from DACN.dacn_cd.cdga.local_search import LocalSearch                 # noqa: E402


def _nx(g):
    G = nx.Graph(); G.add_nodes_from(range(g.n)); G.add_edges_from(zip(g.src.tolist(), g.dst.tolist()))
    return G


def test_objectives_match_networkx():
    g = load_karate()
    G = _nx(g)
    rng = np.random.default_rng(0)
    fac = LARFactory(g, rho=0.5)
    for _ in range(20):
        lab = g.decode(fac.random_individual(rng))
        if lab.max() < 1:
            continue
        q, phi = g.objectives(lab)
        comms = [set(np.flatnonzero(lab == c)) for c in range(lab.max() + 1)]
        assert abs(q - nx.community.modularity(G, comms)) < 1e-10
        phis = [nx.conductance(G, c) for c in comms]
        assert abs(phi - np.mean(phis)) < 1e-10


def test_lar_validity_decode_and_reencode_roundtrip():
    for g in (load_karate(), make_lfr(n=200, mu=0.3, seed=2)):
        fac = LARFactory(g, rho=0.7)
        rng = np.random.default_rng(1)
        for _ in range(30):
            x = fac.random_individual(rng)
            # gen hợp lệ: g_i ∈ N(i)
            assert all(x[i] in g.neighbors(i) for i in range(g.n))
            x = fac.mutate(fac.uniform_crossover(x, fac.random_individual(rng), rng), 0.05, rng)
            assert all(x[i] in g.neighbors(i) for i in range(g.n))
            lab = g.decode(x)
            assert np.bincount(lab).min() >= 2                       # không có singleton
            y = fac.reencode(lab)
            assert all(y[i] in g.neighbors(i) for i in range(g.n))
            assert np.array_equal(canonical(g.decode(y)), canonical(lab))


def test_mutation_changes_gene_to_other_neighbor():
    g = load_karate()
    fac = LARFactory(g, rho=0.5)
    rng = np.random.default_rng(3)
    x = fac.random_individual(rng)
    y = fac.mutate(x, 1.0, rng)
    for i in range(g.n):
        if g.deg[i] > 1:
            assert y[i] != x[i] and y[i] in g.neighbors(i)
        else:
            assert y[i] == x[i]


def test_incremental_delta_matches_full_recompute():
    g = make_lfr(n=200, mu=0.3, seed=2)
    fac = LARFactory(g, rho=0.7)
    rng = np.random.default_rng(5)
    ls = LocalSearch(g, multi=True)
    n_checked = 0
    for _ in range(10):
        lab = g.decode(fac.random_individual(rng))
        K = int(lab.max()) + 1
        if K < 3:
            continue
        l, d = g.community_stats(lab, K)
        size = np.bincount(lab, minlength=K)
        q, phi = g.q_phi_from_stats(l, d)
        for u in rng.permutation(g.n)[:60].tolist():
            for (b, q2, p2, *_rest) in ls.evaluate_moves(lab.tolist(), l.tolist(), d.tolist(),
                                                         size.tolist(), q, phi, u, K):
                new = lab.copy(); new[u] = b
                # phân hoạch mới vẫn có đúng K cộng đồng, mỗi cộng đồng liên thông và >= 2 đỉnh
                assert np.bincount(new, minlength=K).min() >= 2
                G = _nx(g)
                for c in (int(lab[u]), b):
                    assert nx.is_connected(G.subgraph(np.flatnonzero(new == c).tolist()))
                q_full, p_full = g.objectives(new)
                assert abs(q2 - q_full) < 1e-10 and abs(p2 - p_full) < 1e-10
                n_checked += 1
    assert n_checked > 50


def test_local_search_result_is_consistent_and_not_worse_in_pareto_sense():
    g = make_lfr(n=200, mu=0.3, seed=2)
    fac = LARFactory(g, rho=0.7)
    rng = np.random.default_rng(7)
    for multi in (True, False):
        ls = LocalSearch(g, multi=multi)
        for _ in range(5):
            lab0 = g.decode(fac.random_individual(rng))
            if lab0.max() < 1:
                continue
            lab, q, phi, changed = ls.run(lab0, 10, rng)
            q_full, phi_full = g.objectives(lab)
            assert abs(q - q_full) < 1e-9 and abs(phi - phi_full) < 1e-9     # giá trị gia tăng = đầy đủ
            q0, p0 = g.objectives(lab0)
            F0, F1 = pareto.to_min(q0, p0, multi), pareto.to_min(q, phi, multi)
            assert not pareto.dominates(F0, F1)                                # nghiệm trả về không bị C0 trội
            y = fac.reencode(lab)
            assert np.array_equal(canonical(g.decode(y)), canonical(lab))      # Reencode khôi phục đúng
            if multi is False and changed:
                assert q >= q0 - 1e-12


def test_nd_sort_crowding_hv():
    rng = np.random.default_rng(0)
    F = rng.random((60, 2))
    rank = pareto.nd_sort(F)
    for i in range(len(F)):
        dominated_by = [j for j in range(len(F)) if pareto.dominates(F[j], F[i])]
        if rank[i] == 0:
            assert not dominated_by
        else:
            assert max(rank[j] for j in dominated_by) == rank[i] - 1 or max(rank[j] for j in dominated_by) < rank[i]
    # HV: hai điểm, tính tay
    P = np.array([[0.2, 0.6], [0.5, 0.3]])
    expected = (1.1 - 0.2) * (1.1 - 0.6) + (1.1 - 0.5) * (0.6 - 0.3)
    assert abs(pareto.hypervolume(P) - expected) < 1e-12
    # điểm bị trội không làm đổi HV
    P2 = np.vstack([P, [0.6, 0.7]])
    assert abs(pareto.hypervolume(P2) - expected) < 1e-12
    cd = pareto.crowding_distance(np.array([[0, 1], [0.5, 0.5], [1, 0]]))
    assert np.isinf(cd[0]) and np.isinf(cd[2]) and abs(cd[1] - 2.0) < 1e-12


def test_engine_smoke_all_algorithms_karate():
    g = load_karate()
    cfg = Config(mu=30, lam=30, g_max=25, k_stag=25, i_ls=5, p_ls=0.3)
    for algo in ("GA", "HGA", "NSGA2", "MO-HGA-LS"):
        eng = Engine(g, algo, cfg, seed=1)
        arc = eng.run()
        assert len(arc) >= 1
        for s in arc:
            q, phi = g.objectives(s.labels)
            assert abs(q - s.q) < 1e-9 and abs(phi - s.phi) < 1e-9
            assert s.k >= 2
        if algo in ("GA", "HGA"):
            assert len(arc) == 1                       # đơn mục tiêu: chỉ nghiệm Q tốt nhất
        sols = select_solutions(arc, eng.multi)
        assert sols["maxQ"].q >= 0.3


if __name__ == "__main__":
    import inspect
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            fn(); print("ok  ", name)
