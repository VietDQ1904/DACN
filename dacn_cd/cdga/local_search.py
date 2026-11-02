"""Pareto-Lamarck Local Search (Thuật toán 2, mục 2.3).

Toán tử duy nhất: Node Movement trên các đỉnh ranh giới B(C). Các thống kê
(l_c, d_c, |C_c|) được duy trì để ΔQ, ΔΦ tính trong O(deg(u)) mà không quét lại E.

Ràng buộc khả thi của phép chuyển u: C_a -> C_b (mục 2.3.3 và 2.3.6):
  * C_b chứa ít nhất một láng giềng của u (nên C_b + u vẫn liên thông);
  * |C_a| >= 3 (C_a sau khi bỏ u vẫn có >= 2 đỉnh, để Reencode/LAR hợp lệ);
  * C_a \\ {u} vẫn liên thông.
Vì vậy số cộng đồng K KHÔNG đổi trong Local Search và ΔΦ = (Δφ_a + Δφ_b) / K.

Điều kiện dừng sớm (khác Thuật toán 2 ở dòng 15): một vòng lặp ngoài không làm A_L thay đổi
=> dừng. Lý do: chấp nhận cả các bước đi giữa các nghiệm không bị trội có thể quay vòng vô hạn
mà không sinh thêm nghiệm mới; cờ `moved` khi đó không bảo đảm hội tụ.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph, canonical

EPS = 1e-12


def _dominates(a, b) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def _nd(points) -> list[int]:
    """Chỉ số các điểm không bị trội (vài điểm -> O(k^2) thuần Python)."""
    out = []
    for i, p in enumerate(points):
        if not any(_dominates(points[j], p) for j in range(len(points)) if j != i):
            out.append(i)
    return out


def _representative(points, prefer: int = 0) -> int:
    """Điểm gần điểm lý tưởng nhất sau chuẩn hóa min-max; hòa -> ưu tiên `prefer`."""
    if len(points) == 1:
        return 0
    lo0 = min(p[0] for p in points); hi0 = max(p[0] for p in points)
    lo1 = min(p[1] for p in points); hi1 = max(p[1] for p in points)
    s0 = (hi0 - lo0) or 1.0
    s1 = (hi1 - lo1) or 1.0
    dist = [((p[0] - lo0) / s0) ** 2 + ((p[1] - lo1) / s1) ** 2 for p in points]
    best = min(dist)
    if dist[prefer] <= best + EPS:
        return prefer
    return dist.index(best)


class LocalSearch:
    def __init__(self, g: Graph, multi: bool = True, local_archive_max: int = 20):
        self.g = g
        self.multi = multi
        self.al_max = local_archive_max
        self.eval_count = 0     # số phép chuyển được đánh giá gia tăng

    def _F(self, q, phi):
        return (1.0 - q, phi if self.multi else 0.0)

    # ------------------------------------------------------------------
    def _connected_without(self, lab, a, u, nbrs_a) -> bool:
        """C_a \\ {u} có liên thông không?

        C_a liên thông nên mỗi thành phần của C_a \\ {u} chứa ít nhất một láng giềng của u;
        do đó chỉ cần kiểm tra mọi láng giềng của u trong C_a cùng thuộc một thành phần
        (duyệt DFS và dừng sớm khi đã gặp hết).
        """
        adj = self.g.adj_lists
        need = set(nbrs_a[1:])
        seen = {nbrs_a[0]}
        stack = [nbrs_a[0]]
        while stack and need:
            x = stack.pop()
            for y in adj[x]:
                if y != u and lab[y] == a and y not in seen:
                    seen.add(y)
                    need.discard(y)
                    stack.append(y)
        return not need

    def evaluate_moves(self, lab, l, d, size, q, phi, u, K):
        """EvaluateMoves(G, C, u). `lab`, `l`, `d`, `size` là danh sách Python.

        Trả về [(b, q_mới, phi_mới, la', lb', da', db')] cho mọi phép chuyển khả thi.
        """
        g = self.g
        a = lab[u]
        if size[a] < 3:
            return []
        adj_u = g.adj_lists[u]
        counts: dict[int, int] = {}
        for v in adj_u:
            c = lab[v]
            counts[c] = counts.get(c, 0) + 1
        k_ua = counts.get(a, 0)
        if len(counts) == (1 if k_ua else 0):
            return []   # mọi láng giềng cùng cộng đồng: không phải đỉnh ranh giới
        if k_ua >= 2:
            nbrs_a = [v for v in adj_u if lab[v] == a]
            if not self._connected_without(lab, a, u, nbrs_a):
                return []
        m, m2, ku = g.m, 2.0 * g.m, float(g.deg[u])
        la, da = l[a], d[a]
        la_n, da_n = la - k_ua, da - ku
        phi_a = g.conductance_of(la, da)
        phi_a_n = g.conductance_of(la_n, da_n)
        qa_old = la / m - (da / m2) ** 2
        qa_new = la_n / m - (da_n / m2) ** 2
        out = []
        for b, kub in counts.items():
            if b == a:
                continue
            lb, db = l[b], d[b]
            lb_n, db_n = lb + kub, db + ku
            dq = (qa_new - qa_old + lb_n / m - (db_n / m2) ** 2 - lb / m + (db / m2) ** 2)
            dphi = (phi_a_n + g.conductance_of(lb_n, db_n) - phi_a - g.conductance_of(lb, db)) / K
            out.append((b, q + dq, phi + dphi, la_n, lb_n, da_n, db_n))
        self.eval_count += len(out)
        return out

    # ------------------------------------------------------------------
    def run(self, labels0: np.ndarray, i_ls: int, rng: np.random.Generator):
        """Thuật toán 2 (không gồm Reencode). Trả về (labels*, Q*, Φ*, đã_thay_đổi)."""
        g = self.g
        lab_np = labels0.copy()
        lab = lab_np.tolist()
        K = int(lab_np.max()) + 1
        l_np, d_np = g.community_stats(lab_np, K)
        l, d = l_np.tolist(), d_np.tolist()
        size = np.bincount(lab_np, minlength=K).tolist()
        q, phi = g.q_phi_from_stats(l_np, d_np)
        al = [(list(lab), q, phi)]                  # A_L: (nhãn, Q, Φ)
        al_F = [self._F(q, phi)]

        for _ in range(i_ls):
            cross = lab_np[g.src] != lab_np[g.dst]
            boundary = np.unique(np.concatenate([g.src[cross], g.dst[cross]]))
            if len(boundary) == 0:
                break
            al_changed = False
            for u in rng.permutation(boundary).tolist():
                moves = self.evaluate_moves(lab, l, d, size, q, phi, u, K)
                if not moves:
                    continue
                cur = self._F(q, phi)
                pts = [cur] + [self._F(mv[1], mv[2]) for mv in moves]
                nd = _nd(pts)                                         # N* = ND({C} ∪ N)
                # A_L <- ND(A_L ∪ N*)
                for j in nd:
                    if j == 0:
                        continue
                    f = pts[j]
                    if any(_dominates(x, f) or x == f for x in al_F):
                        continue
                    mv = moves[j - 1]
                    new_lab = list(lab)
                    new_lab[u] = mv[0]
                    keep = [t for t, x in enumerate(al_F) if not _dominates(f, x)]
                    al = [al[t] for t in keep] + [(new_lab, mv[1], mv[2])]
                    al_F = [al_F[t] for t in keep] + [f]
                    al_changed = True
                    if len(al) > self.al_max:
                        from . import pareto
                        kp = pareto.truncate_by_crowding(np.array(al_F), self.al_max)
                        al = [al[t] for t in kp]
                        al_F = [al_F[t] for t in kp]
                if nd == [0]:
                    continue                                          # N* \ {C} rỗng
                rep = _representative([pts[j] for j in nd],
                                      prefer=nd.index(0) if 0 in nd else 0)
                j = nd[rep]
                if j == 0:
                    continue
                b, q, phi, la_n, lb_n, da_n, db_n = moves[j - 1]
                a = lab[u]
                l[a], l[b], d[a], d[b] = la_n, lb_n, da_n, db_n
                size[a] -= 1
                size[b] += 1
                lab[u] = b
                lab_np[u] = b
            if not al_changed:
                break

        # C* <- SelectRepresentative(A_L)
        j = _representative(al_F, prefer=0)
        lab_star, q_star, phi_star = al[j]
        lab_star = np.array(lab_star, dtype=np.int32)
        changed = not np.array_equal(lab_star, labels0)
        return (canonical(lab_star) if changed else labels0), q_star, phi_star, changed
