"""Biểu đồ bổ sung cho từng bộ dữ liệu (đi kèm cdga/report_plots.py):

  dataset_overview.png      phân bố bậc, hệ số phân cụm, kích thước cộng đồng chuẩn, bảng thống kê
  anytime.png               Q và Hypervolume theo THỜI GIAN CHẠY; thời gian mỗi thế hệ
  ls_effect.png             hiệu quả của Local Search (tỉ lệ cải thiện, ΔQ, ΔΦ, chi phí mỗi lần gọi)
  network_partitions.png    đồ thị tô màu theo phân hoạch chuẩn / Louvain / Leiden / MO-HGA-LS (n <= 1500)
  contingency.png           ma trận phân hoạch tìm được so với phân hoạch chuẩn
  community_sizes.png       phân bố kích thước cộng đồng: chuẩn so với thuật toán

Các hình dùng nhãn phân hoạch/đồ thị do scripts/run_experiments.py lưu (chỉ khi n <= --save_labels_max_n).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402

from .report_plots import (COLORS, GA_FAMILY, LABEL, _nodata, _save, algos_in, vals)   # noqa: E402

GRAY = (0.78, 0.78, 0.78, 1.0)


# ======================================================================
# Thống kê dữ liệu
# ======================================================================
def fig_dataset_overview(prof: dict, ds: str, out: Path):
    if not prof:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    n = prof["n"]
    # (a) phân bố bậc: CCDF log-log
    ax = axes[0, 0]
    dv, dc = np.array(prof["degree_values"], float), np.array(prof["degree_counts"], float)
    ccdf = dc[::-1].cumsum()[::-1] / n
    ax.loglog(dv, ccdf, "o-", ms=3, lw=1, color="#3182bd")
    ax.axvline(prof["avg_degree"], color="k", ls="--", lw=1)
    ax.text(prof["avg_degree"], ccdf.min() * 1.5, f" ⟨k⟩={prof['avg_degree']:.1f}", fontsize=8)
    ax.set_xlabel("bậc k"); ax.set_ylabel("P(K ≥ k)"); ax.set_title("(a) Phân bố bậc (CCDF, log–log)", fontsize=9)
    # (b) hệ số phân cụm cục bộ
    ax = axes[0, 1]
    h = np.array(prof["cc_hist"], float)
    ax.bar(np.linspace(0.025, 0.975, len(h)), h / h.sum(), width=0.045, color="#31a354", alpha=0.85)
    ax.axvline(prof["avg_cc"], color="k", ls="--", lw=1)
    ax.text(prof["avg_cc"], ax.get_ylim()[1] * 0.95, f" TB={prof['avg_cc']:.3f}\n transitivity={prof['transitivity']:.3f}",
            fontsize=8, va="top")
    ax.set_xlabel("hệ số phân cụm cục bộ CC(i)"); ax.set_ylabel("tỉ lệ đỉnh")
    ax.set_title("(b) Phân bố hệ số phân cụm", fontsize=9)
    # (c) kích thước cộng đồng chuẩn
    ax = axes[1, 0]
    sizes = prof.get("truth_sizes") or []
    if sizes:
        ax.bar(range(1, len(sizes) + 1), sizes, color="#e6550d", alpha=0.85)
        if len(sizes) > 1 and max(sizes) / max(min(sizes), 1) > 30:
            ax.set_yscale("log")
        ax.set_xlabel("hạng cộng đồng chuẩn (theo kích thước)"); ax.set_ylabel("số đỉnh")
        ax.set_title(f"(c) Kích thước {len(sizes)} cộng đồng chuẩn ({prof['n_labeled']}/{n} đỉnh có nhãn)", fontsize=9)
    else:
        _nodata(ax, "(c) Cộng đồng chuẩn")
    # (d) bảng thống kê
    ax = axes[1, 1]
    ax.axis("off")
    raw = prof.get("raw", {})
    rows = [("Số đỉnh gốc → sau tiền xử lý", f"{raw.get('n_raw', n):,} → {n:,}"),
            ("Số cạnh đầu vào", f"{raw.get('m_raw_input', '–'):,}" + (" (có hướng)" if raw.get("directed_raw") or raw.get("m_raw_input", 0) != raw.get("m_raw_undirected") else "")
             if isinstance(raw.get("m_raw_input"), int) else "–"),
            ("Số cạnh vô hướng sau tiền xử lý m", f"{prof['m']:,}"),
            ("Bậc: TB / trung vị / min / max", f"{prof['avg_degree']:.2f} / {prof['median_degree']:.0f} / {prof['min_degree']} / {prof['max_degree']}"),
            ("Mật độ", f"{prof['density']:.5f}"),
            ("CC trung bình / transitivity", f"{prof['avg_cc']:.4f} / {prof['transitivity']:.4f}"),
            ("Số cộng đồng chuẩn K", "–" if prof.get("truth_K") is None else str(prof["truth_K"])),
            ("Tỉ lệ cạnh liên cộng đồng (μ̂)", "–" if prof.get("mixing") is None else f"{prof['mixing']:.3f}"),
            ("Q của phân hoạch chuẩn", "–" if prof.get("truth_Q") is None else f"{prof['truth_Q']:.4f}"),
            ("Φ của phân hoạch chuẩn", "–" if prof.get("truth_Phi") is None else f"{prof['truth_Phi']:.4f}")]
    for i, (k, v) in enumerate(rows):
        y = 0.95 - i * 0.092
        ax.text(0.0, y, k, fontsize=9, va="top", transform=ax.transAxes)
        ax.text(1.0, y, v, fontsize=9, va="top", ha="right", fontweight="bold", transform=ax.transAxes)
        ax.plot([0, 1], [y - 0.06, y - 0.06], color="#ddd", lw=0.6, transform=ax.transAxes)
    ax.set_title("(d) Thống kê (sau tiền xử lý: vô hướng, bỏ self-loop, thành phần liên thông lớn nhất)", fontsize=9)
    fig.suptitle(f"Bộ dữ liệu {ds}")
    fig.tight_layout()
    return _save(fig, out / "dataset_overview.png")


# ======================================================================
# Chất lượng theo thời gian chạy
# ======================================================================
def _curves(sub, algo, ycol):
    rows = sub[sub["algo"].astype(str) == algo]
    T, Y = [], []
    if "time_history" not in rows.columns or ycol not in rows.columns:
        return T, Y
    for t, y in zip(rows["time_history"], rows[ycol]):
        if isinstance(t, list) and isinstance(y, list) and len(t) == len(y) and len(t) > 0:
            T.append(np.asarray(t, float)); Y.append(np.asarray(y, float))
    return T, Y


def _step_on_grid(T, Y, grid):
    """Giá trị tại thời điểm x = giá trị của thế hệ cuối cùng hoàn thành trước x (giữ nguyên sau khi dừng)."""
    out = np.full((len(T), len(grid)), np.nan)
    for i, (t, y) in enumerate(zip(T, Y)):
        idx = np.searchsorted(t, grid, side="right") - 1
        ok = idx >= 0
        out[i, ok] = y[idx[ok]]
    return out


def fig_anytime(sub, ds, out: Path):
    fam = [a for a in algos_in(sub) if a in GA_FAMILY]
    curves = {a: {c: _curves(sub, a, c) for c in ("q_history", "hv_history")} for a in fam}
    fam = [a for a in fam if curves[a]["q_history"][0]]
    if not fam:
        return None
    t_all = [t for a in fam for t in curves[a]["q_history"][0]]
    tmin = max(min(t[0] for t in t_all), 1e-3)
    tmax = max(t[-1] for t in t_all)
    grid = np.geomspace(tmin, tmax, 300) if tmax > tmin * 1.05 else np.linspace(tmin, tmin * 1.05 + 1e-3, 300)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (col, ttl) in zip(axes[:2], (("q_history", "Q lớn nhất trong archive"), ("hv_history", "Hypervolume của archive"))):
        for a in fam:
            T, Y = curves[a][col]
            M = _step_on_grid(T, Y, grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(M, axis=0)
                lo, hi = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
            ax.plot(grid, med, color=COLORS[a], label=LABEL[a])
            ax.fill_between(grid, lo, hi, color=COLORS[a], alpha=0.15)
        for a in ("Louvain", "Leiden"):
            q = vals(sub, a, "maxQ_Q" if col == "q_history" else "hv_final")
            t = vals(sub, a, "time_s")
            if len(q) and len(t):
                ax.axhline(q.mean(), color=COLORS[a], ls="--", lw=1)
                ax.scatter([max(t.mean(), tmin)], [q.mean()], marker="*", s=90, color=COLORS[a], edgecolor="k", zorder=5,
                           label=f"{LABEL[a]} (xong sau {t.mean():.2f}s)")
        ax.set_xscale("log"); ax.set_xlabel("thời gian chạy tích lũy (s, log)"); ax.set_ylabel(ttl)
        ax.set_title(f"{ttl} theo thời gian (trung vị, IQR)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    for a in fam:
        T, _ = curves[a]["q_history"]
        d = [np.diff(t) for t in T if len(t) > 2]
        if not d:
            continue
        L = max(len(x) for x in d)
        M = np.array([np.pad(x, (0, L - len(x)), constant_values=np.nan) for x in d])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(M, axis=0)
        ax.plot(np.arange(1, L + 1), med, color=COLORS[a], label=LABEL[a])
    ax.set_yscale("log"); ax.set_xlabel("thế hệ"); ax.set_ylabel("thời gian mỗi thế hệ (s, log)")
    ax.set_title("Thời gian mỗi thế hệ (trung vị)", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle(f"{ds}: chất lượng nghiệm theo thời gian chạy (anytime performance)")
    fig.tight_layout()
    return _save(fig, out / "anytime.png")


# ======================================================================
# Hiệu quả của Local Search
# ======================================================================
def fig_ls_effect(sub, ds, out: Path):
    need = {"ls_calls", "ls_changed", "ls_improved", "ls_dq", "ls_dphi", "time_ls"}
    if not need <= set(sub.columns):
        return None
    fam = [a for a in ("HGA", "MO-HGA-LS") if len(vals(sub, a, "ls_calls")) and vals(sub, a, "ls_calls").sum() > 0]
    if not fam:
        return None
    d = {}
    for a in fam:
        s = sub[sub["algo"].astype(str) == a]
        c = {k: pd.to_numeric(s[k], errors="coerce").to_numpy(float) for k in need | {"time_s"}}
        with np.errstate(divide="ignore", invalid="ignore"):
            d[a] = {"changed": 100 * c["ls_changed"] / c["ls_calls"], "improved": 100 * c["ls_improved"] / c["ls_calls"],
                    "dq": c["ls_dq"] / c["ls_changed"], "dphi": c["ls_dphi"] / c["ls_changed"],
                    "ms": 1000 * c["time_ls"] / c["ls_calls"], "share": 100 * c["time_ls"] / c["time_s"],
                    "calls": c["ls_calls"]}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    w = 0.36
    ax = axes[0]
    for j, (k, lab, col) in enumerate((("changed", "LS đổi phân hoạch", "#6baed6"), ("improved", "nghiệm sau LS trội nghiệm trước", "#2171b5"))):
        mu = [np.nanmean(d[a][k]) for a in fam]
        sd = [np.nanstd(d[a][k], ddof=1) if len(d[a][k]) > 1 else 0 for a in fam]
        ax.bar(np.arange(len(fam)) + (j - 0.5) * w, mu, w, yerr=sd, capsize=3, color=col, label=lab)
        for i, v in enumerate(mu):
            ax.text(i + (j - 0.5) * w, v + (sd[i] if np.isfinite(sd[i]) else 0) + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(fam))); ax.set_xticklabels([LABEL[a] for a in fam]); ax.set_ylabel("% số lần gọi LS")
    ax.set_ylim(0, 118); ax.set_title("(a) Tỉ lệ LS có tác dụng", fontsize=9); ax.legend(fontsize=7, loc="upper left", ncol=1)
    for ax, k, ttl in ((axes[1], "dq", "(b) ΔQ trung bình mỗi lần LS đổi phân hoạch"),
                       (axes[2], "dphi", "(c) ΔΦ trung bình mỗi lần LS đổi phân hoạch")):
        mu = [np.nanmean(d[a][k]) for a in fam]
        sd = [np.nanstd(d[a][k], ddof=1) if len(d[a][k]) > 1 else 0 for a in fam]
        ax.bar(range(len(fam)), mu, yerr=sd, capsize=3, color=[COLORS[a] for a in fam])
        ax.axhline(0, color="k", lw=0.8)
        for i, v in enumerate(mu):
            ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_xticks(range(len(fam))); ax.set_xticklabels([LABEL[a] for a in fam]); ax.set_title(ttl, fontsize=9)
    ax = axes[3]
    mu = [np.nanmean(d[a]["ms"]) for a in fam]
    ax.bar(range(len(fam)), mu, color=[COLORS[a] for a in fam])
    for i, a in enumerate(fam):
        ax.text(i, mu[i], f"{mu[i]:.1f} ms\n{np.nanmean(d[a]['share']):.0f}% tổng thời gian\n{np.nanmean(d[a]['calls']):.0f} lần gọi/lần chạy",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(mu) * 1.45 if mu else 1)
    ax.set_xticks(range(len(fam))); ax.set_xticklabels([LABEL[a] for a in fam]); ax.set_ylabel("ms")
    ax.set_title("(d) Chi phí mỗi lần gọi Local Search", fontsize=9)
    fig.suptitle(f"{ds}: hiệu quả của Local Search (mỗi lần LS = một lần áp dụng lên một cá thể con)")
    fig.tight_layout()
    return _save(fig, out / "ls_effect.png")


# ======================================================================
# Hình dựa trên nhãn phân hoạch
# ======================================================================
def _best_row(sub, algo):
    s = sub[sub["algo"].astype(str) == algo]
    if "labels" not in s.columns or s.empty:
        return None
    s = s[s["labels"].map(lambda x: isinstance(x, dict))]
    q = pd.to_numeric(s.get("maxQ_Q"), errors="coerce")
    if s.empty or q.isna().all():
        return None
    return s.loc[q.idxmax()]


def _labels_of(row, key):
    if row is None or key not in row["labels"]:
        return None
    return np.asarray(row["labels"][key], dtype=np.int64)


def _truth_colors(truth):
    cmap = plt.get_cmap("tab20")
    ids, cnt = np.unique(truth[truth >= 0], return_counts=True)
    order = ids[np.argsort(-cnt, kind="stable")]
    return {int(c): cmap(i % 20) for i, c in enumerate(order)}


def _node_colors(labels, truth=None, tcol=None):
    """Tô màu: cộng đồng tìm được lấy màu của lớp chuẩn trùng nhiều nhất (tham lam) để các hình dễ so sánh."""
    cmap = plt.get_cmap("tab20")
    ids, cnt = np.unique(labels[labels >= 0], return_counts=True)
    order = [int(c) for c in ids[np.argsort(-cnt, kind="stable")]]
    colmap, used = {}, set()
    if truth is not None and tcol and (truth >= 0).any():
        ok = (truth >= 0) & (labels >= 0)
        pairs = {}
        for f, t in zip(labels[ok].tolist(), truth[ok].tolist()):
            pairs[(f, t)] = pairs.get((f, t), 0) + 1
        for (f, t), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if f not in colmap and t not in used and t in tcol:
                colmap[f] = tcol[t]; used.add(t)
    spare = [cmap(i) for i in range(20) if cmap(i) not in colmap.values()]
    k = 0
    for c in order:
        if c not in colmap:
            colmap[c] = spare[k] if k < len(spare) else GRAY
            k += 1
    return np.array([colmap.get(int(c), GRAY) if c >= 0 else (0.92, 0.92, 0.92, 1.0) for c in labels])


def fig_network(sub, ds, out: Path, gpath):
    if gpath is None or not Path(gpath).exists():
        return None
    import networkx as nx
    z = np.load(gpath)
    n = int(z["n"])
    if n > 1500:
        return None
    truth = np.asarray(z["truth"], dtype=np.int64)
    has_truth = bool((truth >= 0).any())
    panels = []
    if has_truth:
        panels.append(("Phân hoạch chuẩn (ground truth)", truth, None))
    for a, key in (("Louvain", "maxQ"), ("Leiden", "maxQ"), ("HGA", "maxQ"), ("MO-HGA-LS", "maxQ"), ("MO-HGA-LS", "knee")):
        row = _best_row(sub, a)
        lab = _labels_of(row, key)
        if lab is not None:
            extra = f", NMI={row[f'{key}_NMI']:.3f}" if f"{key}_NMI" in row and pd.notna(row.get(f"{key}_NMI")) else ""
            panels.append((f"{LABEL[a]} – nghiệm {key}\nK={int(lab.max()) + 1}, Q={row[f'{key}_Q']:.3f}, Φ={row[f'{key}_Phi']:.3f}{extra}", lab, a))
    if len(panels) < 2:
        return None
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(z["src"].tolist(), z["dst"].tolist()))
    pos = nx.spring_layout(G, seed=7, iterations=100)
    xy = np.array([pos[i] for i in range(n)])
    deg = np.array([G.degree(i) for i in range(n)], float)
    size = 6 + 38 * np.sqrt(deg / deg.max()) * (1.0 if n < 400 else 0.55)
    segs = np.stack([xy[z["src"]], xy[z["dst"]]], axis=1)
    alpha = float(np.clip(40.0 / np.sqrt(max(len(segs), 1)), 0.04, 0.5))
    tcol = _truth_colors(truth) if has_truth else None
    cols = 3 if len(panels) > 4 or len(panels) == 3 else 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (ttl, lab, _a) in zip(axes, panels):
        ax.add_collection(LineCollection(segs, colors="#777777", linewidths=0.35, alpha=alpha, zorder=1))
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=_node_colors(lab, truth if has_truth else None, tcol),
                   edgecolors="k", linewidths=0.25, zorder=2)
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(xy[:, 0].min() - 0.05, xy[:, 0].max() + 0.05); ax.set_ylim(xy[:, 1].min() - 0.05, xy[:, 1].max() + 0.05)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{ds}: phân hoạch tìm được (lần chạy có Q cao nhất của mỗi thuật toán; màu cộng đồng khớp màu lớp chuẩn gần nhất)")
    fig.tight_layout()
    return _save(fig, out / "network_partitions.png")


def _contingency(truth, found, max_cols=25):
    m = (truth >= 0) & (found >= 0)
    t, f = truth[m], found[m]
    tk = np.unique(t)
    fk, fcnt = np.unique(f, return_counts=True)
    fk = fk[np.argsort(-fcnt, kind="stable")]
    tcnt = {c: int((t == c).sum()) for c in tk}
    tk = sorted(tk, key=lambda c: -tcnt[c])
    keep = list(fk[:max_cols])
    M = np.zeros((len(tk), len(keep) + (1 if len(fk) > max_cols else 0)), int)
    tpos = {c: i for i, c in enumerate(tk)}
    fpos = {c: i for i, c in enumerate(keep)}
    for ti, fi in zip(t.tolist(), f.tolist()):
        M[tpos[ti], fpos.get(fi, M.shape[1] - 1)] += 1
    return M, len(fk) > max_cols


def fig_contingency(sub, ds, out: Path, gpath):
    if gpath is None or not Path(gpath).exists():
        return None
    z = np.load(gpath)
    truth = np.asarray(z["truth"], dtype=np.int64)
    if not (truth >= 0).any() or truth.max() + 1 > 60:
        return None
    items = []
    for a in ("MO-HGA-LS", "HGA", "Louvain", "Leiden"):
        row = _best_row(sub, a)
        lab = _labels_of(row, "maxQ")
        if lab is not None:
            items.append((a, row, lab))
    items = items[:3]
    if not items:
        return None
    fig, axes = plt.subplots(1, len(items), figsize=(5.6 * len(items), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (a, row, lab) in zip(axes, items):
        M, merged = _contingency(truth, lab)
        R = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        if M.size <= 300:
            for (i, j), v in np.ndenumerate(M):
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="w" if R[i, j] > 0.55 else "k")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([str(i + 1) for i in range(M.shape[1] - (1 if merged else 0))] + (["khác"] if merged else []), fontsize=7)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"lớp {i + 1}" for i in range(M.shape[0])], fontsize=7)
        ax.set_xlabel("cộng đồng tìm được (theo kích thước giảm dần)"); ax.grid(False)
        nmi = row.get("maxQ_NMI")
        ax.set_title(f"{LABEL[a]}: K={int(lab.max()) + 1}" + (f", NMI={nmi:.3f}" if pd.notna(nmi) else ""), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.045, label="tỉ lệ đỉnh của lớp chuẩn")
    fig.suptitle(f"{ds}: ma trận nhầm lẫn giữa lớp chuẩn (hàng) và cộng đồng tìm được (cột); số = số đỉnh")
    fig.tight_layout()
    return _save(fig, out / "contingency.png")


def fig_community_sizes(sub, ds, out: Path, prof):
    series = []
    sizes = (prof or {}).get("truth_sizes") or []
    if sizes:
        series.append(("Phân hoạch chuẩn", np.array(sizes), "k"))
    for a in algos_in(sub):
        row = _best_row(sub, a)
        lab = _labels_of(row, "maxQ")
        if lab is not None:
            series.append((f"{LABEL[a]} (K={int(lab.max()) + 1})", np.sort(np.bincount(lab))[::-1], COLORS[a]))
    if len(series) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for name, s, col in series:
        s = s[s > 0]
        ax.plot(np.arange(1, len(s) + 1), s, "o-" if name.startswith("Phân hoạch") else ".-", ms=4, lw=1.3, color=col, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("hạng cộng đồng (theo kích thước, log)"); ax.set_ylabel("số đỉnh (log)")
    ax.set_title(f"{ds}: phân bố kích thước cộng đồng", fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out / "community_sizes.png")
