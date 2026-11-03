"""Biểu đồ phân tích cho từng bộ dữ liệu và tổng hợp liên bộ dữ liệu (Chương 3).

Đầu vào: DataFrame do scripts/analyze.py::load_runs tạo ra (mỗi dòng là một lần chạy).
Mọi hàm đều chịu được dữ liệu thiếu (ví dụ bộ dữ liệu không có ground-truth, chỉ chạy baseline).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from scipy import stats           # noqa: E402

from . import pareto              # noqa: E402

ORDER = ["GA", "HGA", "NSGA2", "MO-HGA-LS", "Louvain", "Leiden"]
LABEL = {"GA": "GA", "HGA": "HGA (GA+LS)", "NSGA2": "NSGA-II", "MO-HGA-LS": "MO-HGA-LS",
         "Louvain": "Louvain", "Leiden": "Leiden"}
COLORS = {"GA": "#8c8c8c", "HGA": "#e69f00", "NSGA2": "#56b4e9", "MO-HGA-LS": "#d55e00",
          "Louvain": "#009e73", "Leiden": "#cc79a7"}
GA_FAMILY = ("GA", "HGA", "NSGA2", "MO-HGA-LS")
MULTI = ("NSGA2", "MO-HGA-LS")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def algos_in(sub) -> list[str]:
    have = set(sub["algo"].astype(str))
    return [a for a in ORDER if a in have]


def vals(sub, algo, col) -> np.ndarray:
    if col not in sub.columns:
        return np.array([])
    s = sub.loc[sub["algo"].astype(str) == algo, col]
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _nodata(ax, title):
    ax.set_title(title, fontsize=9)
    ax.text(0.5, 0.5, "không có dữ liệu", ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def _xticks(ax, algos):
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([LABEL[a].replace(" (", "\n(") for a in algos], fontsize=7)


def box_panel(ax, sub, algos, col, title, log=False, hline=None, hline_label=None):
    data = [vals(sub, a, col) for a in algos]
    if not any(len(d) for d in data):
        return _nodata(ax, title)
    rng = np.random.default_rng(0)
    for i, (a, d) in enumerate(zip(algos, data)):
        if len(d) == 0:
            continue
        bp = ax.boxplot([d], positions=[i], widths=0.55, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="k", markersize=4),
                        medianprops=dict(color="k"), flierprops=dict(markersize=0))
        bp["boxes"][0].set(facecolor=COLORS[a], alpha=0.55)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=8, color=COLORS[a], edgecolor="k", linewidth=0.3, zorder=3)
    if hline is not None and not np.isnan(hline):
        ax.axhline(hline, color="k", ls="--", lw=1)
        ax.text(len(algos) - 0.5, hline, hline_label or "chuẩn", ha="right", va="bottom", fontsize=7)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    _xticks(ax, algos)
    ax.set_xlim(-0.6, len(algos) - 0.4)


def bar_panel(ax, sub, algos, col, title, fmt="{:.3f}"):
    m = [vals(sub, a, col) for a in algos]
    if not any(len(d) for d in m):
        return _nodata(ax, title)
    means = [d.mean() if len(d) else np.nan for d in m]
    sds = [d.std(ddof=1) if len(d) > 1 else 0 for d in m]
    ax.bar(range(len(algos)), np.nan_to_num(means), yerr=sds, capsize=3, color=[COLORS[a] for a in algos], alpha=0.85)
    for i, (mu, sd) in enumerate(zip(means, sds)):
        if not np.isnan(mu):
            ax.text(i, mu + sd, fmt.format(mu), ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=9)
    _xticks(ax, algos)


# ======================================================================
# Biểu đồ theo từng bộ dữ liệu
# ======================================================================
def fig_dashboard(sub, ds, out):
    algos = algos_in(sub)
    t_k = pd.to_numeric(sub.get("truth_K"), errors="coerce").dropna()
    truth_k = float(t_k.iloc[0]) if len(t_k) else np.nan
    t_q = pd.to_numeric(sub.get("truth_Q"), errors="coerce").dropna()
    truth_q = float(t_q.iloc[0]) if len(t_q) else np.nan
    t_p = pd.to_numeric(sub.get("truth_Phi"), errors="coerce").dropna()
    truth_p = float(t_p.iloc[0]) if len(t_p) else np.nan
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    box_panel(axes[0, 0], sub, algos, "maxQ_Q", "Modularity Q (↑)", hline=truth_q, hline_label="phân hoạch chuẩn")
    box_panel(axes[0, 1], sub, algos, "maxQ_Phi", "Conductance trung bình Φ (↓)", hline=truth_p, hline_label="phân hoạch chuẩn")
    box_panel(axes[0, 2], sub, algos, "maxQ_NMI", "NMI (↑)")
    box_panel(axes[0, 3], sub, algos, "maxQ_ARI", "ARI (↑)")
    box_panel(axes[1, 0], sub, algos, "time_s", "Thời gian chạy (s, thang log)", log=True)
    box_panel(axes[1, 1], sub, algos, "maxQ_K", "Số cộng đồng K", hline=truth_k, hline_label="K chuẩn")
    box_panel(axes[1, 2], sub, algos, "hv_final", "Hypervolume của archive (↑)")
    box_panel(axes[1, 3], sub, [a for a in algos if a in GA_FAMILY], "n_eval", "Số lần đánh giá hàm mục tiêu")
    n_runs = sub.groupby(sub["algo"].astype(str)).size().to_dict()
    row0 = sub.iloc[0]
    fig.suptitle(f"{ds}: n = {int(row0['n'])}, m = {int(row0['m'])}  |  số lần chạy: "
                 + ", ".join(f"{LABEL[a]} {n_runs[a]}" for a in algos), fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "dashboard.png")


def fig_bars(sub, ds, out):
    algos = algos_in(sub)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (c, t) in zip(axes, (("maxQ_Q", "Q (↑)"), ("maxQ_Phi", "Φ (↓)"), ("maxQ_NMI", "NMI (↑)"), ("maxQ_ARI", "ARI (↑)"))):
        bar_panel(ax, sub, algos, c, f"{t} – trung bình ± độ lệch chuẩn")
    fig.suptitle(f"{ds}: các chỉ số đánh giá (nghiệm Q lớn nhất của mỗi lần chạy)")
    fig.tight_layout()
    return _save(fig, out / "bars_metrics.png")


def _truth_point(sub):
    q = pd.to_numeric(sub.get("truth_Q"), errors="coerce").dropna()
    p = pd.to_numeric(sub.get("truth_Phi"), errors="coerce").dropna()
    return (float(q.iloc[0]), float(p.iloc[0])) if len(q) and len(p) else None


def fig_pareto(sub, ds, out):
    algos = algos_in(sub)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    tp = _truth_point(sub)
    # (a) mọi lần chạy
    ax = axes[0]
    for a in algos:
        s = sub[sub["algo"].astype(str) == a]
        if a in MULTI:
            pts = np.vstack([np.array(f) for f in s["front"] if len(f)])
            ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.28, color=COLORS[a], label=f"{LABEL[a]} (mọi nghiệm trong archive)")
        else:
            ax.scatter(vals(s, a, "maxQ_Q"), vals(s, a, "maxQ_Phi"), s=22, alpha=0.7, color=COLORS[a],
                       edgecolor="k", linewidth=0.3, label=LABEL[a])
    if tp:
        ax.scatter(*tp, marker="*", s=160, color="k", zorder=5, label="phân hoạch chuẩn")
    ax.set_title("(a) Nghiệm của tất cả các lần chạy", fontsize=9)
    ax.set_xlabel("Modularity Q (↑)"); ax.set_ylabel("Conductance trung bình Φ (↓)")
    ax.legend(fontsize=6.5)
    # (b) front tổng hợp (ND của mọi lần chạy) và nghiệm đại diện
    ax = axes[1]
    for a in algos:
        s = sub[sub["algo"].astype(str) == a]
        if a in MULTI:
            pts = np.vstack([np.array(f) for f in s["front"] if len(f)])
            F = pareto.to_min(pts[:, 0], pts[:, 1], True)
            nd = pareto.nd_indices(F)
            P = pts[nd][np.argsort(pts[nd][:, 0])]
            ax.plot(P[:, 0], P[:, 1], "-o", ms=3, lw=1.2, color=COLORS[a], label=f"{LABEL[a]}: front tổng hợp ({len(P)} điểm)")
            for key, mk in (("knee", "s"), ("minPhi", "v"), ("maxQ", "^")):
                if f"{key}_Q" in s:
                    ax.scatter(vals(s, a, f"{key}_Q").mean(), vals(s, a, f"{key}_Phi").mean(), marker=mk, s=45,
                               color=COLORS[a], edgecolor="k", zorder=4)
        else:
            q, p = vals(s, a, "maxQ_Q"), vals(s, a, "maxQ_Phi")
            if len(q):
                ax.errorbar(q.mean(), p.mean(), xerr=q.std(), yerr=p.std(), fmt="o", color=COLORS[a], capsize=3, label=f"{LABEL[a]} (TB ± sd)")
    if tp:
        ax.scatter(*tp, marker="*", s=160, color="k", zorder=5)
    ax.set_title("(b) Front tổng hợp; ▲ maxQ, ■ knee, ▼ minΦ (trung bình)", fontsize=9)
    ax.set_xlabel("Modularity Q (↑)")
    ax.legend(fontsize=6.5)
    fig.suptitle(f"{ds}: Pareto front Q – Φ")
    fig.tight_layout()
    return _save(fig, out / "pareto.png")


def _hist(sub, algo, col):
    if col not in sub.columns:
        return None
    hs = [h for h in sub.loc[sub["algo"].astype(str) == algo, col] if isinstance(h, list) and len(h)]
    if not hs:
        return None
    L = max(len(h) for h in hs)
    return np.array([h + [h[-1]] * (L - len(h)) for h in hs], dtype=float)


def fig_convergence(sub, ds, out):
    algos = [a for a in algos_in(sub) if a in GA_FAMILY]
    panels = (("hv_history", "Hypervolume của archive"), ("q_history", "Q lớn nhất trong archive"),
              ("archive_size_history", "Kích thước archive"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    drawn = False
    for ax, (col, ttl) in zip(axes, panels):
        for a in algos:
            M = _hist(sub, a, col)
            if M is None or (col == "archive_size_history" and a in ("GA", "HGA")):
                continue
            x = np.arange(M.shape[1])
            ax.plot(x, np.median(M, axis=0), color=COLORS[a], label=LABEL[a])
            ax.fill_between(x, np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0), color=COLORS[a], alpha=0.15)
            drawn = True
        ax.set_title(ttl + " (trung vị, IQR)", fontsize=9)
        ax.set_xlabel("Thế hệ")
        ax.legend(fontsize=7)
    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle(f"{ds}: hội tụ theo thế hệ")
    fig.tight_layout()
    return _save(fig, out / "convergence.png")


def fig_time(sub, ds, out):
    algos = algos_in(sub)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) tổng thời gian, thang log
    ax = axes[0]
    means = [vals(sub, a, "time_s").mean() for a in algos]
    sds = [vals(sub, a, "time_s").std(ddof=1) if len(vals(sub, a, "time_s")) > 1 else 0 for a in algos]
    ax.bar(range(len(algos)), means, yerr=sds, capsize=3, color=[COLORS[a] for a in algos], alpha=0.85)
    for i, mu in enumerate(means):
        ax.text(i, mu, f"{mu:.2f}s" if mu < 100 else f"{mu:.0f}s", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log"); ax.set_title("(a) Tổng thời gian chạy (thang log)", fontsize=9); _xticks(ax, algos)
    # (b) tỉ lệ thời gian Local Search / phần còn lại
    ax = axes[1]
    fam = [a for a in algos if a in GA_FAMILY]
    if fam and "time_ls" in sub.columns:
        ls = [vals(sub, a, "time_ls").mean() if len(vals(sub, a, "time_ls")) else 0.0 for a in fam]
        tot = [vals(sub, a, "time_s").mean() for a in fam]
        rest = [t - l for t, l in zip(tot, ls)]
        ax.bar(range(len(fam)), rest, color="#9ecae1", label="GA (giải mã, đánh giá, chọn lọc)")
        ax.bar(range(len(fam)), ls, bottom=rest, color="#e6550d", label="Local Search + Reencode")
        for i, (r, l) in enumerate(zip(rest, ls)):
            if r + l > 0:
                ax.text(i, r + l, f"LS {100 * l / (r + l):.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(fam))); ax.set_xticklabels([LABEL[a].replace(" (", "\n(") for a in fam], fontsize=7)
        ax.legend(fontsize=7)
        ax.set_ylabel("giây")
    else:
        _nodata(ax, "(b)")
    ax.set_title("(b) Cơ cấu thời gian của họ GA", fontsize=9)
    # (c) số lần đánh giá
    ax = axes[2]
    if fam:
        w = 0.38
        ne = [vals(sub, a, "n_eval").mean() if len(vals(sub, a, "n_eval")) else 0 for a in fam]
        nl = [vals(sub, a, "n_ls_eval").mean() if len(vals(sub, a, "n_ls_eval")) else 0 for a in fam]
        ax.bar(np.arange(len(fam)) - w / 2, ne, w, color="#3182bd", label="đánh giá đầy đủ (giải mã + Q, Φ)")
        ax.bar(np.arange(len(fam)) + w / 2, nl, w, color="#e6550d", label="phép chuyển LS đánh giá gia tăng")
        ax.set_yscale("log")
        ax.set_xticks(range(len(fam))); ax.set_xticklabels([LABEL[a].replace(" (", "\n(") for a in fam], fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("(c) Số lần đánh giá hàm mục tiêu (log)", fontsize=9)
    fig.suptitle(f"{ds}: chi phí tính toán")
    fig.tight_layout()
    return _save(fig, out / "time_cost.png")


def fig_quality_vs_time(sub, ds, out):
    algos = algos_in(sub)
    metrics = [("maxQ_Q", "Q (↑)"), ("maxQ_NMI", "NMI (↑)"), ("hv_final", "Hypervolume (↑)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (c, ttl) in zip(axes, metrics):
        drew = False
        for a in algos:
            t, y = vals(sub, a, "time_s"), vals(sub, a, c)
            if len(t) == 0 or len(y) == 0:
                continue
            ax.errorbar(t.mean(), y.mean(), xerr=t.std(), yerr=y.std(), fmt="o", ms=8, color=COLORS[a], capsize=3, label=LABEL[a])
            drew = True
        if not drew:
            _nodata(ax, ttl); continue
        ax.set_xscale("log"); ax.set_xlabel("Thời gian trung bình (s, log)"); ax.set_ylabel(ttl)
        ax.set_title(f"{ttl} so với thời gian", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{ds}: chất lượng – chi phí")
    fig.tight_layout()
    return _save(fig, out / "quality_vs_time.png")


def fig_q_vs_nmi(sub, ds, out):
    if "maxQ_NMI" not in sub.columns or sub["maxQ_NMI"].isna().all():
        return None
    algos = algos_in(sub)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, c in zip(axes, ("maxQ_NMI", "maxQ_ARI")):
        for a in algos:
            s = sub[sub["algo"].astype(str) == a]
            ax.scatter(pd.to_numeric(s["maxQ_Q"]), pd.to_numeric(s[c]), s=22, alpha=0.75, color=COLORS[a], edgecolor="k", linewidth=0.3, label=LABEL[a])
        x = pd.to_numeric(sub["maxQ_Q"]); y = pd.to_numeric(sub[c])
        ok = x.notna() & y.notna()
        if ok.sum() > 3 and x[ok].std() > 0:
            r = stats.spearmanr(x[ok], y[ok])[0]
            ax.text(0.03, 0.95, f"Spearman ρ = {r:.2f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("Q"); ax.set_ylabel(c.split("_")[1]); ax.set_title(f"Q và {c.split('_')[1]} (mỗi chấm là một lần chạy)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{ds}: Modularity cao có khớp phân hoạch chuẩn không?")
    fig.tight_layout()
    return _save(fig, out / "q_vs_nmi.png")


def fig_ls_ablation(sub, ds, out):
    pairs = [("GA", "HGA", "đơn mục tiêu"), ("NSGA2", "MO-HGA-LS", "đa mục tiêu")]
    pairs = [p for p in pairs if len(vals(sub, p[0], "maxQ_Q")) and len(vals(sub, p[1], "maxQ_Q"))]
    if not pairs:
        return None
    cols = [("maxQ_Q", "Q (↑)", False), ("hv_final", "Hypervolume (↑)", False), ("maxQ_NMI", "NMI (↑)", False), ("time_s", "Thời gian (s)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    w = 0.36
    for ax, (c, ttl, _) in zip(axes, cols):
        if not any(len(vals(sub, a, c)) for p in pairs for a in p[:2]):
            _nodata(ax, ttl); continue
        for i, (a, b, desc) in enumerate(pairs):
            va, vb = vals(sub, a, c), vals(sub, b, c)
            for j, (alg, v) in enumerate(((a, va), (b, vb))):
                if len(v):
                    ax.bar(i + (j - 0.5) * w, v.mean(), w, yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=3,
                           color=COLORS[alg], alpha=0.9, label=LABEL[alg] if i == 0 or True else None)
            if len(va) and len(vb) and va.mean() != 0:
                ax.text(i + 0.5 * w, vb.mean(), f"{100 * (vb.mean() / va.mean() - 1):+.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{LABEL[a]} → {LABEL[b]}\n({d})" for a, b, d in pairs], fontsize=7)
        ax.set_title(ttl, fontsize=9)
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=6.5)
    fig.suptitle(f"{ds}: ảnh hưởng của Local Search (% = thay đổi tương đối khi thêm LS)")
    fig.tight_layout()
    return _save(fig, out / "ls_ablation.png")


def fig_solutions(sub, ds, out):
    """Các nghiệm đại diện maxQ / knee / minΦ của thuật toán đa mục tiêu."""
    algos = [a for a in MULTI if len(vals(sub, a, "knee_Q"))]
    if not algos:
        return None
    mets = [("Q", "Q (↑)"), ("Phi", "Φ (↓)"), ("NMI", "NMI (↑)"), ("ARI", "ARI (↑)"), ("K", "K")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    w = 0.8 / len(algos)
    for ax, (m, ttl) in zip(axes, mets):
        if not any(len(vals(sub, a, f"{k}_{m}")) for a in algos for k in ("maxQ", "knee", "minPhi")):
            _nodata(ax, ttl); continue
        for j, a in enumerate(algos):
            mu = [vals(sub, a, f"{k}_{m}").mean() if len(vals(sub, a, f"{k}_{m}")) else np.nan for k in ("maxQ", "knee", "minPhi")]
            sd = [vals(sub, a, f"{k}_{m}").std(ddof=1) if len(vals(sub, a, f"{k}_{m}")) > 1 else 0 for k in ("maxQ", "knee", "minPhi")]
            ax.bar(np.arange(3) + (j - (len(algos) - 1) / 2) * w, np.nan_to_num(mu), w, yerr=sd, capsize=2, color=COLORS[a], alpha=0.9, label=LABEL[a])
        ax.set_xticks(range(3)); ax.set_xticklabels(["max Q", "knee", "min Φ"], fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=6.5)
    fig.suptitle(f"{ds}: ba nghiệm đại diện của tập Pareto (quy tắc chọn nghiệm cuối cùng, mục 2.5.4e)")
    fig.tight_layout()
    return _save(fig, out / "representative_solutions.png")


def paired_values(sub, a, b, col):
    keys = ["seed"]
    if "_config_key" in sub.columns:
        keys.insert(0, "_config_key")
    sa = sub[sub["algo"].astype(str) == a].drop_duplicates(keys).set_index(keys)[col] if col in sub else None
    sb = sub[sub["algo"].astype(str) == b].drop_duplicates(keys).set_index(keys)[col] if col in sub else None
    if sa is None or sb is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    common = sa.index.intersection(sb.index)
    x = pd.to_numeric(sa.loc[common], errors="coerce")
    y = pd.to_numeric(sb.loc[common], errors="coerce")
    ok = x.notna() & y.notna()
    return x[ok].to_numpy(float), y[ok].to_numpy(float)


def pairwise_wilcoxon(sub, algos, col):
    P = np.full((len(algos), len(algos)), np.nan)
    D = np.full_like(P, np.nan)
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i == j:
                continue
            x, y = paired_values(sub, a, b, col)
            if len(x) < 6:
                continue
            D[i, j] = (x - y).mean()
            P[i, j] = 1.0 if np.all(x == y) else stats.wilcoxon(x, y).pvalue
    return P, D


def fig_wilcoxon(sub, ds, out):
    algos = algos_in(sub)
    if len(algos) < 2:
        return None
    cols = [(c, t) for c, t in (("maxQ_Q", "Q"), ("maxQ_NMI", "NMI"), ("hv_final", "Hypervolume")) if c in sub.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.6))
    axes = np.atleast_1d(axes)
    any_ok = False
    for ax, (c, t) in zip(axes, cols):
        P, D = pairwise_wilcoxon(sub, algos, c)
        if np.all(np.isnan(P)):
            _nodata(ax, f"{t}"); continue
        any_ok = True
        im = ax.imshow(np.where(np.isnan(P), np.nan, np.clip(P, 0, 0.2)), cmap="RdYlGn_r", vmin=0, vmax=0.2)
        for i in range(len(algos)):
            for j in range(len(algos)):
                if not np.isnan(P[i, j]):
                    star = "*" if P[i, j] < 0.05 else ""
                    ax.text(j, i, f"{P[i, j]:.3g}{star}\nΔ={D[i, j]:+.3f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(algos))); ax.set_yticks(range(len(algos)))
        ax.set_xticklabels([LABEL[a].split(" (")[0] for a in algos], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([LABEL[a].split(" (")[0] for a in algos], fontsize=7)
        ax.grid(False); ax.set_title(f"{t}: p của Wilcoxon (hàng so với cột; * p<0,05)", fontsize=8)
    if not any_ok:
        plt.close(fig); return None
    fig.suptitle(f"{ds}: kiểm định Wilcoxon signed-rank ghép cặp theo seed (cần ≥ 6 cặp)")
    fig.tight_layout()
    return _save(fig, out / "wilcoxon.png")


def fig_norm_heatmap(sub, ds, out):
    """Tổng quan: mỗi chỉ số chuẩn hóa min–max qua các thuật toán (1 = tốt nhất)."""
    algos = algos_in(sub)
    spec = [("maxQ_Q", "Q", True), ("maxQ_Phi", "Φ", False), ("maxQ_NMI", "NMI", True), ("maxQ_ARI", "ARI", True),
            ("hv_final", "HV", True), ("time_s", "Tốc độ (log thời gian)", False)]
    rows, names = [], []
    for c, nm, hi in spec:
        mu = np.array([vals(sub, a, c).mean() if len(vals(sub, a, c)) else np.nan for a in algos])
        if np.all(np.isnan(mu)):
            continue
        if c == "time_s":
            mu = np.log10(np.maximum(mu, 1e-6))
        z = (mu - np.nanmin(mu)) / ((np.nanmax(mu) - np.nanmin(mu)) or 1.0)
        rows.append(z if hi else 1 - z); names.append(nm)
    # ổn định: 1 - hệ số biến thiên của Q (chuẩn hóa)
    cv = np.array([vals(sub, a, "maxQ_Q").std(ddof=1) / abs(vals(sub, a, "maxQ_Q").mean()) if len(vals(sub, a, "maxQ_Q")) > 1 else np.nan for a in algos])
    if not np.all(np.isnan(cv)):
        z = (cv - np.nanmin(cv)) / ((np.nanmax(cv) - np.nanmin(cv)) or 1.0)
        rows.append(1 - z); names.append("Ổn định (CV của Q)")
    if not rows:
        return None
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(1.0 * len(algos) + 3, 0.55 * len(names) + 1.6))
    im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(algos))); ax.set_xticklabels([LABEL[a].replace(" (", "\n(") for a in algos], fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.grid(False)
    ax.set_title(f"{ds}: điểm tương đối (1 = tốt nhất trong các thuật toán, 0 = kém nhất)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return _save(fig, out / "relative_scores.png")


def per_dataset_figures(sub: pd.DataFrame, ds: str, out_root: Path, profile: dict | None = None,
                        graph_path=None) -> list[Path]:
    """Vẽ mọi biểu đồ của một bộ dữ liệu vào <out_root>/<ds>/.

    profile    : hồ sơ thống kê (datasets.json) -> dataset_overview, community_sizes
    graph_path : tệp .npz (cạnh + nhãn chuẩn) -> network_partitions, contingency
    """
    from . import report_plots2 as rp2
    out = out_root / safe(ds)
    jobs = [("dashboard", fig_dashboard), ("bars", fig_bars), ("pareto", fig_pareto), ("convergence", fig_convergence),
            ("time_cost", fig_time), ("quality_vs_time", fig_quality_vs_time), ("q_vs_nmi", fig_q_vs_nmi),
            ("ls_ablation", fig_ls_ablation), ("solutions", fig_solutions), ("wilcoxon", fig_wilcoxon),
            ("relative_scores", fig_norm_heatmap), ("anytime", rp2.fig_anytime), ("ls_effect", rp2.fig_ls_effect),
            ("dataset_overview", lambda s, d, o: rp2.fig_dataset_overview(profile, d, o)),
            ("network_partitions", lambda s, d, o: rp2.fig_network(s, d, o, graph_path)),
            ("contingency", lambda s, d, o: rp2.fig_contingency(s, d, o, graph_path)),
            ("community_sizes", lambda s, d, o: rp2.fig_community_sizes(s, d, o, profile))]
    made = []
    for nm, fn in jobs:
        try:
            p = fn(sub, ds, out)
            if p:
                made.append(p)
        except Exception as e:   # noqa: BLE001 - một hình lỗi không làm hỏng cả báo cáo
            print(f"  [cảnh báo] {nm} cho {ds}: {type(e).__name__}: {e}")
    return made


# ======================================================================
# Tổng hợp liên bộ dữ liệu
# ======================================================================
def summary_figures(df: pd.DataFrame, out_root: Path) -> list[Path]:
    out = out_root / "summary"
    made = []
    dss = list(dict.fromkeys(df["dataset"].astype(str)))
    algos = algos_in(df)
    spec = [("maxQ_Q", "Modularity Q (↑)", False), ("maxQ_NMI", "NMI (↑)", False), ("maxQ_ARI", "ARI (↑)", False),
            ("time_s", "Thời gian chạy (s, log)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (c, ttl, log) in zip(axes.ravel(), spec):
        w = 0.8 / len(algos)
        any_v = False
        for j, a in enumerate(algos):
            mu, sd = [], []
            for ds in dss:
                v = vals(df[df["dataset"].astype(str) == ds], a, c)
                mu.append(v.mean() if len(v) else np.nan); sd.append(v.std(ddof=1) if len(v) > 1 else 0)
            if not np.all(np.isnan(mu)):
                any_v = True
                ax.bar(np.arange(len(dss)) + (j - (len(algos) - 1) / 2) * w, np.nan_to_num(mu), w, yerr=sd, capsize=1.5,
                       color=COLORS[a], alpha=0.9, label=LABEL[a])
        if not any_v:
            _nodata(ax, ttl); continue
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(len(dss))); ax.set_xticklabels(dss, fontsize=8)
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("So sánh trên mọi bộ dữ liệu (trung bình ± độ lệch chuẩn)")
    fig.tight_layout()
    made.append(_save(fig, out / "all_datasets_metrics.png"))

    # hạng trung bình theo Q (và NMI) qua các bộ dữ liệu
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (c, ttl) in zip(axes, (("maxQ_Q", "Q"), ("maxQ_NMI", "NMI"))):
        ranks = {a: [] for a in algos}
        for ds in dss:
            mu = {a: vals(df[df["dataset"].astype(str) == ds], a, c) for a in algos}
            mu = {a: v.mean() for a, v in mu.items() if len(v)}
            if len(mu) < 2:
                continue
            order = sorted(mu, key=lambda a: -mu[a])
            for r, a in enumerate(order, 1):
                ranks[a].append(r)
        have = [a for a in algos if ranks[a]]
        if not have:
            _nodata(ax, f"Hạng theo {ttl}"); continue
        ax.bar(range(len(have)), [np.mean(ranks[a]) for a in have], color=[COLORS[a] for a in have])
        for i, a in enumerate(have):
            ax.text(i, np.mean(ranks[a]), f"{np.mean(ranks[a]):.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(have))); ax.set_xticklabels([LABEL[a].replace(" (", "\n(") for a in have], fontsize=7)
        ax.set_title(f"Hạng trung bình theo {ttl} (1 = tốt nhất)", fontsize=9)
    fig.tight_layout()
    made.append(_save(fig, out / "mean_rank.png"))

    # thời gian theo kích thước đồ thị
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for a in algos:
        xs, ys = [], []
        for ds in dss:
            s = df[df["dataset"].astype(str) == ds]
            v = vals(s, a, "time_s")
            if len(v):
                xs.append(float(s["m"].iloc[0])); ys.append(v.mean())
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=COLORS[a], label=LABEL[a])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số cạnh m"); ax.set_ylabel("Thời gian trung bình (s)")
    ax.set_title("Thời gian chạy theo quy mô đồ thị"); ax.legend(fontsize=7)
    fig.tight_layout()
    made.append(_save(fig, out / "time_vs_size.png"))
    return made
