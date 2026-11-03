"""Động cơ tiến hóa dùng chung cho 4 biến thể (Chương 2 và đề cương Chương 3):

  algo        mục tiêu        Local Search   Ghi chú
  ----------  --------------  -------------  -----------------------------------
  GA          Q (đơn)         không          GA + LAR, cùng tham số chính
  HGA         Q (đơn)         có (p_LS)      MA-Q / HGA-LS đơn mục tiêu
  NSGA2       Q và Φ (Pareto) không          baseline MOEA (= MO-HGA-LS khi p_LS = 0)
  MO-HGA-LS   Q và Φ (Pareto) có (p_LS)      phương pháp đề xuất (Thuật toán 1)

Chế độ đơn mục tiêu dùng vectơ F = (1-Q, 0): toàn bộ cơ chế Pareto/NSGA-II suy
biến thành xếp hạng theo Q, nên dùng chung mã nguồn với chế độ đa mục tiêu.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from . import pareto
from .graph import Graph, partition_key, canonical
from .lar import LARFactory
from .local_search import LocalSearch

ALGOS = ("GA", "HGA", "NSGA2", "MO-HGA-LS")


@dataclass
class Config:
    mu: int = 100            # kích thước quần thể cha
    lam: int = 100           # số cá thể con mỗi thế hệ (λ)
    pc: float = 0.85         # xác suất lai ghép
    pm: float | None = None  # xác suất đột biến mỗi gen; None -> 1/n
    tournament: int = 2      # t
    i_ls: int = 30           # I_LS
    p_ls: float = 0.2        # xác suất áp dụng Local Search (cần hiệu chỉnh)
    rho: float = 0.7         # tỉ lệ khởi tạo thiên lệch theo CC (cần hiệu chỉnh)
    a_max: int = 100         # kích thước tối đa của Pareto archive
    g_max: int = 200         # số thế hệ tối đa
    k_stag: int = 30         # số thế hệ trì trệ liên tiếp
    eps_stop: float = 1e-4   # ngưỡng cải thiện HV tối thiểu

    def to_dict(self):
        return asdict(self)


@dataclass
class Solution:
    labels: np.ndarray
    q: float
    phi: float

    @property
    def k(self) -> int:
        return int(self.labels.max()) + 1


class Engine:
    def __init__(self, g: Graph, algo: str, cfg: Config, seed: int = 0):
        assert algo in ALGOS, algo
        self.g, self.algo, self.cfg = g, algo, cfg
        self.multi = algo in ("NSGA2", "MO-HGA-LS")
        self.use_ls = algo in ("HGA", "MO-HGA-LS") and cfg.p_ls > 0
        self.rng = np.random.default_rng(seed)
        self.pm = cfg.pm if cfg.pm is not None else 1.0 / g.n
        self.fac = LARFactory(g, cfg.rho)
        self.ls = LocalSearch(g, multi=self.multi)
        self.n_eval = 0          # số lần đánh giá đầy đủ hàm mục tiêu
        self.hv_history: list[float] = []
        self.q_history: list[float] = []
        self.t_ls = 0.0          # tổng thời gian trong Local Search + Reencode
        self.t_history: list[float] = []   # thời gian tích lũy (s) tại cuối mỗi thế hệ (phần tử đầu: sau khởi tạo)
        self.ls_calls = 0        # số lần Local Search được gọi
        self.ls_changed = 0      # số lần LS trả về phân hoạch khác phân hoạch ban đầu
        self.ls_improved = 0     # số lần nghiệm sau LS trội nghiệm trước LS (theo mục tiêu đang tối ưu)
        self.ls_dq = 0.0         # tổng ΔQ (sau - trước) trên các lần LS có thay đổi
        self.ls_dphi = 0.0       # tổng ΔΦ
        self.archive_size_history: list[int] = []

    # ------------------------------------------------------------------
    def _F(self, q, phi):
        return pareto.to_min(q, phi, self.multi)

    def _evaluate(self, genes: np.ndarray):
        labels = self.g.decode(genes)
        self.n_eval += 1
        q, phi = self.g.objectives(labels)
        return labels, q, phi

    # ------------------------------------------------------------------
    def _generate_offspring(self, genes, rank, crowd):
        cfg, rng = self.cfg, self.rng
        mu = len(genes)

        def tournament(k):
            cand = rng.integers(0, mu, size=(k, cfg.tournament))
            best = cand[:, 0].copy()
            for t in range(1, cfg.tournament):
                c = cand[:, t]
                better = (rank[c] < rank[best]) | ((rank[c] == rank[best]) & (crowd[c] > crowd[best]))
                best = np.where(better, c, best)
            return best

        p1 = tournament(cfg.lam)
        p2 = tournament(cfg.lam)
        out_genes = np.empty((cfg.lam, self.g.n), dtype=np.int32)
        for i in range(cfg.lam):
            if rng.random() < cfg.pc:
                child = self.fac.uniform_crossover(genes[p1[i]], genes[p2[i]], rng)
            else:
                child = genes[p1[i]].copy()
            out_genes[i] = self.fac.mutate(child, self.pm, rng)
        return out_genes

    def _improve(self, genes_i, labels, q, phi):
        """Áp dụng Local Search (Lamarck) với xác suất p_LS."""
        if not (self.use_ls and self.rng.random() < self.cfg.p_ls):
            return genes_i, labels, q, phi
        if int(labels.max()) + 1 < 2:      # K = 1: phân hoạch không hợp lệ (Φ không xác định), không cải thiện
            return genes_i, labels, q, phi
        t0 = time.perf_counter()
        self.ls_calls += 1
        new_labels, q2, phi2, changed = self.ls.run(labels, self.cfg.i_ls, self.rng)
        if not changed:
            self.t_ls += time.perf_counter() - t0
            return genes_i, labels, q, phi          # giữ nguyên kiểu gen gốc
        new_genes = self.fac.reencode(new_labels)    # ghi ngược về LAR
        self.t_ls += time.perf_counter() - t0
        self.ls_changed += 1
        self.ls_dq += q2 - q
        self.ls_dphi += phi2 - phi
        if pareto.dominates(self._F(q2, phi2), self._F(q, phi)):
            self.ls_improved += 1
        return new_genes, new_labels, q2, phi2

    # ------------------------------------------------------------------
    def run(self, verbose: int = 0):
        cfg, g = self.cfg, self.g
        t0 = time.perf_counter()
        # --- khởi tạo ---
        genes = self.fac.init_population(cfg.mu, self.rng)
        labels, Q, PHI = [], [], []
        for x in genes:
            lb, q, p = self._evaluate(x)
            labels.append(lb); Q.append(q); PHI.append(p)
        labels = np.array(labels); Q = np.array(Q); PHI = np.array(PHI)
        F = self._F(Q, PHI)
        rank, crowd = pareto.rank_and_crowding(F)

        archive = self._update_archive([], labels, Q, PHI)
        hv_best = self._hv(archive)
        self.hv_history.append(hv_best)
        self.q_history.append(max(s.q for s in archive))
        self.archive_size_history.append(len(archive))
        self.t_history.append(time.perf_counter() - t0)
        stag, gen = 0, 0

        while gen < cfg.g_max and stag < cfg.k_stag:
            # --- sinh con ---
            off_genes = self._generate_offspring(genes, rank, crowd)
            off_labels = np.empty_like(off_genes)
            oq, op = np.zeros(cfg.lam), np.zeros(cfg.lam)
            for i in range(cfg.lam):
                lb, q, p = self._evaluate(off_genes[i])
                off_genes[i], lb, q, p = self._improve(off_genes[i], lb, q, p)
                off_labels[i], oq[i], op[i] = lb, q, p
            # --- hợp nhất R = P ∪ O, loại trùng phân hoạch, NSGA-II selection ---
            R_genes = np.vstack([genes, off_genes])
            R_labels = np.vstack([labels, off_labels])
            R_Q = np.concatenate([Q, oq]); R_P = np.concatenate([PHI, op])
            seen, keep = set(), []
            for i in range(len(R_genes)):
                key = partition_key(R_labels[i])
                if key not in seen:
                    seen.add(key); keep.append(i)
            keep = np.array(keep)
            if len(keep) < cfg.mu:                       # bù cá thể mới nếu thiếu
                need = cfg.mu - len(keep)
                new_g = self.fac.init_population(need, self.rng)
                ng, nl, nq, npp = [], [], [], []
                for x in new_g:
                    lb, q, p = self._evaluate(x)
                    ng.append(x); nl.append(lb); nq.append(q); npp.append(p)
                R_genes = np.vstack([R_genes[keep], np.array(ng)])
                R_labels = np.vstack([R_labels[keep], np.array(nl)])
                R_Q = np.concatenate([R_Q[keep], nq]); R_P = np.concatenate([R_P[keep], npp])
            else:
                R_genes, R_labels, R_Q, R_P = R_genes[keep], R_labels[keep], R_Q[keep], R_P[keep]
            RF = self._F(R_Q, R_P)
            sel, rank, crowd = pareto.nsga_select(RF, cfg.mu)
            genes, labels, Q, PHI = R_genes[sel], R_labels[sel], R_Q[sel], R_P[sel]

            # --- archive: A <- Truncate(ND(A ∪ O), A_max) ---
            archive = self._update_archive(archive, off_labels, oq, op)
            hv = self._hv(archive)
            if hv > hv_best + cfg.eps_stop:
                hv_best, stag = hv, 0
            else:
                stag += 1
            self.hv_history.append(hv)
            self.q_history.append(max(s.q for s in archive))
            self.archive_size_history.append(len(archive))
            self.t_history.append(time.perf_counter() - t0)
            gen += 1
            if verbose and gen % verbose == 0:
                best_q = max(s.q for s in archive)
                print(f"  gen {gen:>4}  |A|={len(archive):>3}  HV={hv:.4f}  Qmax={best_q:.4f}  "
                      f"stag={stag}  t={time.perf_counter() - t0:.1f}s", flush=True)

        self.time = time.perf_counter() - t0
        self.generations = gen
        self.archive = archive
        return archive

    # ------------------------------------------------------------------
    def _hv(self, archive: list[Solution]) -> float:
        if not archive:
            return 0.0
        F = self._F([s.q for s in archive], [s.phi for s in archive])
        return pareto.hypervolume(F)

    def _update_archive(self, archive, labels, Q, PHI):
        """ND(A ∪ O) rồi cắt theo crowding distance xuống A_max; bỏ trùng vectơ mục tiêu."""
        pool = list(archive)
        have = {(round(s.q, 12), round(s.phi if self.multi else 0.0, 12)) for s in pool}
        for lb, q, p in zip(labels, Q, PHI):
            if int(lb.max()) + 1 < 2:
                continue
            key = (round(float(q), 12), round(float(p) if self.multi else 0.0, 12))
            if key in have:
                continue
            have.add(key)
            pool.append(Solution(np.array(lb, copy=True), float(q), float(p)))
        F = self._F([s.q for s in pool], [s.phi for s in pool])
        nd = pareto.nd_indices(F)
        pool = [pool[i] for i in nd]
        F = F[nd]
        if len(pool) > self.cfg.a_max:
            kp = pareto.truncate_by_crowding(F, self.cfg.a_max)
            pool = [pool[i] for i in kp]
        return pool


def select_solutions(archive: list[Solution], multi: bool) -> dict[str, Solution]:
    """Quy tắc chọn nghiệm cuối cùng (mục 2.5.4e): max Q, min Φ, knee."""
    Qs = np.array([s.q for s in archive]); Ps = np.array([s.phi for s in archive])
    out = {"maxQ": archive[int(np.argmax(Qs))]}
    if multi:
        out["minPhi"] = archive[int(np.argmin(Ps))]
        F = pareto.to_min(Qs, Ps, True)
        out["knee"] = archive[pareto.knee_index(F)]
    return out
