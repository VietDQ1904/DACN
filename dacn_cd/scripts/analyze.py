"""Tổng hợp kết quả thành bảng và hình cho Chương 3 (mục 3.2 - 3.4).

  python scripts/analyze.py --runs results/main/runs.jsonl [--sweep results/sweep_ls/runs.jsonl] --out results/report

Có thể gộp nhiều tệp: --runs results/main/runs.jsonl results/amazon/runs.jsonl

Đầu ra:
  <out>/tables.md, <out>/tables/*.csv      mọi bảng, đánh số theo mục của Chương 3
  <out>/summary_long.csv                   trung bình/độ lệch chuẩn/trung vị mọi chỉ số, mọi (bộ dữ liệu, thuật toán)
  <out>/figures/<bộ dữ liệu>/*.png         mỗi bộ dữ liệu ~17 hình: thống kê dữ liệu, dashboard (Q, Φ, NMI, ARI, K, thời gian, HV),
                                           Pareto, hội tụ, anytime, chi phí, Local Search, Wilcoxon, đồ thị phân hoạch, ...
  <out>/figures/summary/*.png              so sánh liên bộ dữ liệu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdga import report_plots as rp   # noqa: E402
from cdga.report_plots import LABEL, ORDER, safe   # noqa: E402

DROP = ("solutions", "front", "hv_history", "q_history", "archive_size_history", "time_history", "labels", "config")


# ----------------------------------------------------------------------
def load_runs(paths) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                flat = {k: v for k, v in r.items() if k not in DROP}
                flat["_config_key"] = json.dumps(r.get("config") or {}, sort_keys=True, separators=(",", ":"))
                for sol, d in r["solutions"].items():
                    for k, v in d.items():
                        flat[f"{sol}_{k}"] = v
                for k in ("front", "hv_history", "q_history", "archive_size_history", "time_history"):
                    flat[k] = r.get(k, [])
                flat["labels"] = r.get("labels")        # dict {nghiệm: nhãn} hoặc None (đồ thị lớn)
                rows.append(flat)
    df = pd.DataFrame(rows)
    df["algo"] = pd.Categorical(df["algo"], [a for a in ORDER if a in set(df["algo"])], ordered=True)
    return df


def load_profiles(paths):
    """datasets.json và graphs/*.npz nằm cạnh mỗi runs.jsonl."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    profiles, graphs = {}, {}
    for p in paths:
        d = Path(p).parent
        if (d / "datasets.json").exists():
            profiles.update(json.loads((d / "datasets.json").read_text(encoding="utf-8")))
        for ds in list(profiles):
            f = d / "graphs" / f"{safe(ds)}.npz"
            if f.exists():
                graphs[ds] = f
    return profiles, graphs


def ms(x, d=4):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "–"
    return f"{x.mean():.{d}f} ± {x.std(ddof=1) if len(x) > 1 else 0:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


class Report:
    def __init__(self, out: Path):
        self.out = out
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        self.md = ["# Kết quả thực nghiệm (tự động sinh bởi scripts/analyze.py)\n"]

    def table(self, name, title, df, note=""):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")   # utf-8-sig: Excel đọc đúng tiếng Việt
        self.md += [f"\n## {title}\n", md_table(df), f"\n{note}\n" if note else ""]

    def save(self):
        (self.out / "tables.md").write_text("\n".join(self.md), encoding="utf-8")


def pivot_ms(df, metric, sol="maxQ", d=4):
    col = f"{sol}_{metric}"
    rows = []
    for ds, sub in df.groupby("dataset", observed=True):
        row = {"Bộ dữ liệu": ds}
        for a in [x for x in ORDER if x in set(sub["algo"])]:
            s = sub[sub["algo"] == a]
            if col in s:
                row[LABEL[a]] = ms(s[col], d)
        rows.append(row)
    return pd.DataFrame(rows)


def best_marker(tab: pd.DataFrame, higher=True) -> pd.DataFrame:
    tab = tab.copy()
    for i, r in tab.iterrows():
        vals = {}
        for c in tab.columns[1:]:
            try:
                vals[c] = float(str(r[c]).split("±")[0])
            except ValueError:
                pass
        if vals:
            best = (max if higher else min)(vals, key=vals.get)
            tab.loc[i, best] = f"**{r[best]}**"
    return tab


# ----------------------------------------------------------------------
def sec_32(df, rep):
    groups = {"3_2_1_GA": ["GA", "NSGA2"], "3_2_2_HybridGA": ["HGA", "MO-HGA-LS"], "3_2_3_Louvain_Leiden": ["Louvain", "Leiden"]}
    titles = {"3_2_1_GA": "3.2.1 Kết quả của Genetic Algorithm (GA đơn mục tiêu và NSGA-II, đều không có Local Search)",
              "3_2_2_HybridGA": "3.2.2 Kết quả của Hybrid Genetic Algorithm (HGA đơn mục tiêu và MO-HGA-LS)",
              "3_2_3_Louvain_Leiden": "3.2.3 Kết quả của Louvain và Leiden"}
    for key, algos in groups.items():
        rows = []
        for ds, sub in df.groupby("dataset", observed=True):
            for a in algos:
                s = sub[sub["algo"] == a]
                if s.empty:
                    continue
                for sol in ("maxQ", "knee", "minPhi"):
                    if f"{sol}_Q" not in s or s[f"{sol}_Q"].isna().all():
                        continue
                    rows.append({"Bộ dữ liệu": ds, "Thuật toán": LABEL[a], "Nghiệm": sol,
                                 "Q": ms(s[f"{sol}_Q"]), "Φ": ms(s[f"{sol}_Phi"]), "K": ms(s[f"{sol}_K"], 1),
                                 "NMI": ms(s[f"{sol}_NMI"]) if f"{sol}_NMI" in s else "–",
                                 "ARI": ms(s[f"{sol}_ARI"]) if f"{sol}_ARI" in s else "–",
                                 "Thời gian (s)": ms(s["time_s"], 2), "Số lần chạy": len(s)})
        if rows:
            rep.table(f"t{key}", titles[key], pd.DataFrame(rows),
                      "Nghiệm: maxQ = Q lớn nhất trong archive; knee = gần điểm lý tưởng; minPhi = Φ nhỏ nhất. Trung bình ± độ lệch chuẩn.")


def sec_33(df, rep):
    mo = df[df["algo"] == "MO-HGA-LS"]
    rep.table("t3_3_1_modularity", "3.3.1 So sánh Modularity (nghiệm Q lớn nhất của mỗi lần chạy)", best_marker(pivot_ms(df, "Q"), True))
    if not mo.empty:
        rows = []
        for ds, sub in mo.groupby("dataset", observed=True):
            for sol in ("maxQ", "knee", "minPhi"):
                rows.append({"Bộ dữ liệu": ds, "Nghiệm của MO-HGA-LS": sol, "Q": ms(sub[f"{sol}_Q"]),
                             "Φ": ms(sub[f"{sol}_Phi"]), "K": ms(sub[f"{sol}_K"], 1)})
        rep.table("t3_3_1b_mo_trade", "3.3.1b Đánh đổi Q – Φ giữa các nghiệm đại diện của MO-HGA-LS", pd.DataFrame(rows))
    if "maxQ_NMI" in df and df["maxQ_NMI"].notna().any():
        for met in ("NMI", "ARI"):
            t = pivot_ms(df[df["maxQ_NMI"].notna()], met)
            if not t.empty:
                rep.table(f"t3_3_2_{met.lower()}", f"3.3.2 So sánh {met} (nghiệm Q lớn nhất)", best_marker(t, True))
        if not mo.empty and mo["knee_NMI"].notna().any():
            rows = []
            for ds, sub in mo.groupby("dataset", observed=True):
                if sub["knee_NMI"].notna().any():
                    rows.append({"Bộ dữ liệu": ds, **{f"{sol} {met}": ms(sub[f"{sol}_{met}"])
                                                       for sol in ("maxQ", "knee", "minPhi") for met in ("NMI", "ARI")}})
            rep.table("t3_3_2b_mo", "3.3.2b NMI/ARI theo nghiệm đại diện của MO-HGA-LS", pd.DataFrame(rows))
    rows = []
    for ds, sub in df.groupby("dataset", observed=True):
        for a in [x for x in ORDER if x in set(sub["algo"])]:
            s = sub[sub["algo"] == a]
            ls = s["time_ls"] if "time_ls" in s else pd.Series(dtype=float)
            share = (100 * ls.dropna().sum() / s.loc[ls.notna(), "time_s"].sum()) if ls.notna().any() else np.nan
            rows.append({"Bộ dữ liệu": ds, "Thuật toán": LABEL[a], "Thời gian (s)": ms(s["time_s"], 2),
                         "Tỉ lệ thời gian LS (%)": "–" if np.isnan(share) else f"{share:.1f}",
                         "Số thế hệ": ms(s["generations"], 1), "Số lần đánh giá đầy đủ": ms(s["n_eval"], 0),
                         "Số phép chuyển LS đã đánh giá": ms(s["n_ls_eval"], 0)})
    rep.table("t3_3_3_time", "3.3.3 So sánh thời gian thực thi và chi phí tính toán", pd.DataFrame(rows),
              "Louvain/Leiden không có khái niệm thế hệ nên các cột đó để trống.")
    rows = []
    for ds, sub in df.groupby("dataset", observed=True):
        for a in [x for x in ORDER if x in set(sub["algo"])]:
            q = sub[sub["algo"] == a]["maxQ_Q"].dropna()
            hv = sub[sub["algo"] == a]["hv_final"].dropna()
            rows.append({"Bộ dữ liệu": ds, "Thuật toán": LABEL[a], "Q trung vị": f"{q.median():.4f}",
                         "Độ lệch chuẩn": f"{q.std(ddof=1) if len(q) > 1 else 0:.4f}",
                         "Hệ số biến thiên (%)": f"{100 * q.std(ddof=1) / q.mean() if len(q) > 1 else 0:.2f}",
                         "IQR": f"{q.quantile(.75) - q.quantile(.25):.4f}", "Q min": f"{q.min():.4f}", "Q max": f"{q.max():.4f}",
                         "HV trung bình": f"{hv.mean():.4f}", "Số lần chạy": len(q)})
    rep.table("t3_3_4_stability", "3.3.4a Độ ổn định của kết quả qua các lần chạy độc lập", pd.DataFrame(rows))
    rows = []
    for ds, sub in df.groupby("dataset", observed=True):
        if not (sub["algo"] == "MO-HGA-LS").any():
            continue
        for a in [x for x in ORDER if x in set(sub["algo"]) and x != "MO-HGA-LS"]:
            for metric in ("maxQ_Q", "hv_final") + (("maxQ_NMI",) if "maxQ_NMI" in df else ()):
                x, y = rp.paired_values(sub, "MO-HGA-LS", a, metric)
                if len(x) < 6:
                    continue
                diff = x - y
                try:
                    p = stats.wilcoxon(x, y).pvalue if (diff != 0).any() else 1.0
                except ValueError:
                    p = float("nan")
                rows.append({"Bộ dữ liệu": ds, "So sánh": f"MO-HGA-LS vs {LABEL[a]}", "Chỉ số": metric.replace("maxQ_", ""),
                             "n cặp": len(x), "Hiệu trung bình": f"{diff.mean():+.4f}", "p (Wilcoxon signed-rank)": f"{p:.4g}",
                             "Có ý nghĩa (p<0.05)": "có" if p < 0.05 else "không"})
    if rows:
        rep.table("t3_3_4b_wilcoxon", "3.3.4b Kiểm định Wilcoxon signed-rank (ghép cặp theo seed)", pd.DataFrame(rows),
                  "Ghép cặp theo seed chỉ có ý nghĩa tương đối vì các thuật toán dùng luồng ngẫu nhiên khác nhau; cần ≥ 20 lần chạy.")


def sec_34(df, rep):
    pairs = [("GA", "HGA", "đơn mục tiêu (Q)"), ("NSGA2", "MO-HGA-LS", "đa mục tiêu (Q, Φ)")]
    rows = []
    for ds, sub in df.groupby("dataset", observed=True):
        for a, b, desc in pairs:
            sa, sb = sub[sub["algo"] == a], sub[sub["algo"] == b]
            if sa.empty or sb.empty:
                continue
            for name, s in ((f"{LABEL[a]} (không LS)", sa), (f"{LABEL[b]} (có LS)", sb)):
                rows.append({"Bộ dữ liệu": ds, "Bài toán": desc, "Phương pháp": name, "Q": ms(s["maxQ_Q"]), "HV": ms(s["hv_final"]),
                             "NMI": ms(s["maxQ_NMI"]) if "maxQ_NMI" in s else "–", "Thế hệ": ms(s["generations"], 1),
                             "Thời gian (s)": ms(s["time_s"], 2)})
    if rows:
        rep.table("t3_4_1_ls_vs_nols", "3.4.1 So sánh GA và GA + Local Search", pd.DataFrame(rows))


def sec_datasets(profiles, df, rep):
    """Bảng 2.3 cập nhật bằng số liệu thật sau tiền xử lý."""
    rows = []
    for ds in dict.fromkeys(df["dataset"].astype(str)):
        p = profiles.get(ds)
        if not p:
            continue
        raw = p.get("raw", {})
        rows.append({"Bộ dữ liệu": ds, "Đỉnh gốc": raw.get("n_raw", ""), "Cạnh đầu vào": raw.get("m_raw_input", ""),
                     "Có hướng": "có" if raw.get("directed_raw") else "không", "n": p["n"], "m": p["m"],
                     "Bậc TB": f"{p['avg_degree']:.2f}", "Bậc max": p["max_degree"], "Mật độ": f"{p['density']:.5f}",
                     "CC TB": f"{p['avg_cc']:.4f}", "Transitivity": f"{p['transitivity']:.4f}",
                     "K chuẩn": p["truth_K"] if p.get("truth_K") is not None else "–",
                     "Đỉnh có nhãn": f"{p['n_labeled']}/{p['n']}" if p.get("truth_K") is not None else "–",
                     "μ̂ (cạnh liên cộng đồng)": f"{p['mixing']:.3f}" if p.get("mixing") is not None else "–",
                     "Q chuẩn": f"{p['truth_Q']:.4f}" if p.get("truth_Q") is not None else "–",
                     "Φ chuẩn": f"{p['truth_Phi']:.4f}" if p.get("truth_Phi") is not None else "–"})
    if rows:
        rep.table("t2_3_datasets", "Bảng 2.3 (cập nhật) Thông số thống kê các bộ dữ liệu sau tiền xử lý", pd.DataFrame(rows),
                  "Tiền xử lý: vô hướng hóa, bỏ self-loop và cạnh trùng, chỉ giữ thành phần liên thông lớn nhất. "
                  "Q, Φ chuẩn chỉ tính khi mọi đỉnh đều có nhãn. μ̂ = tỉ lệ cạnh nối hai cộng đồng chuẩn khác nhau.")


def sec_per_dataset(df, rep):
    """Mỗi bộ dữ liệu một bảng đầy đủ các chỉ số (phụ lục)."""
    for ds, sub in df.groupby("dataset", observed=True):
        rows = []
        for a in [x for x in ORDER if x in set(sub["algo"])]:
            s = sub[sub["algo"] == a]
            tl = pd.to_numeric(s["time_ls"], errors="coerce") if "time_ls" in s else pd.Series(dtype=float)
            share = 100 * tl.sum() / s.loc[tl.notna(), "time_s"].sum() if tl.notna().any() else np.nan
            rows.append({"Thuật toán": LABEL[a], "Số lần chạy": len(s), "Q": ms(s["maxQ_Q"]), "Φ": ms(s["maxQ_Phi"]),
                         "K": ms(s["maxQ_K"], 1),
                         "NMI": ms(s["maxQ_NMI"]) if "maxQ_NMI" in s else "–", "ARI": ms(s["maxQ_ARI"]) if "maxQ_ARI" in s else "–",
                         "HV": ms(s["hv_final"]), "|Archive|": ms(s["archive_size"], 1), "Thời gian (s)": ms(s["time_s"], 2),
                         "Thế hệ": ms(s["generations"], 1), "Đánh giá đầy đủ": ms(s["n_eval"], 0),
                         "% thời gian LS": "–" if np.isnan(share) else f"{share:.1f}"})
        rep.table(f"dataset_{safe(ds)}", f"Phụ lục – {ds}: toàn bộ chỉ số theo thuật toán (nghiệm Q lớn nhất; trung bình ± độ lệch chuẩn)",
                  pd.DataFrame(rows))


def summary_long(df, out: Path):
    cols = [c for c in df.columns if c.endswith(("_Q", "_Phi", "_K", "_NMI", "_ARI")) or c in
            ("time_s", "time_ls", "hv_final", "generations", "n_eval", "n_ls_eval", "archive_size")]
    rows = []
    for (ds, a), s in df.groupby(["dataset", "algo"], observed=True):
        r = {"dataset": ds, "algo": a, "runs": len(s)}
        for c in cols:
            v = pd.to_numeric(s[c], errors="coerce").dropna()
            if len(v):
                r.update({f"{c}_mean": v.mean(), f"{c}_std": v.std(ddof=1) if len(v) > 1 else 0.0, f"{c}_median": v.median()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary_long.csv", index=False, encoding="utf-8-sig")


def sec_34_sweep(sw, rep, out):
    sw = sw[sw["algo"] == "MO-HGA-LS"].copy()
    if sw.empty:
        return
    for k in ("p_ls", "i_ls"):
        sw[k] = sw["tag"].str.extract(rf"{k}=([\d.]+)").astype(float)
    for ds, sub in sw.groupby("dataset"):
        agg = sub.groupby(["p_ls", "i_ls"]).agg(Q=("maxQ_Q", "mean"), Q_sd=("maxQ_Q", "std"), HV=("hv_final", "mean"),
                                                  t=("time_s", "mean"), gens=("generations", "mean"), n=("seed", "count")).reset_index()
        t = agg.copy()
        t["Q"] = [f"{a:.4f} ± {0 if np.isnan(b) else b:.4f}" for a, b in zip(agg["Q"], agg["Q_sd"])]
        t = t.drop(columns="Q_sd").rename(columns={"p_ls": "p_LS", "i_ls": "I_LS", "t": "Thời gian (s)", "gens": "Thế hệ", "n": "Số lần chạy"})
        t["HV"] = t["HV"].map("{:.4f}".format); t["Thời gian (s)"] = t["Thời gian (s)"].map("{:.2f}".format); t["Thế hệ"] = t["Thế hệ"].map("{:.1f}".format)
        rep.table(f"t3_4_2_sweep_{safe(ds)}", f"3.4.2 Ảnh hưởng của p_LS và I_LS ({ds})", t)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (col, ttl) in zip(axes, (("Q", "Q trung bình"), ("HV", "HV trung bình"), ("t", "Thời gian (s)"))):
            pv = agg.pivot(index="p_ls", columns="i_ls", values=col)
            im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(pv.columns)), [f"{c:g}" for c in pv.columns]); ax.set_yticks(range(len(pv.index)), [f"{c:g}" for c in pv.index])
            ax.set_xlabel("$I_{LS}$"); ax.set_ylabel("$p_{LS}$"); ax.set_title(ttl); ax.grid(False)
            for (i, j), v in np.ndenumerate(pv.values):
                ax.text(j, i, f"{v:.3g}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Khảo sát tham số Local Search – {ds}"); fig.tight_layout()
        d = out / "figures" / safe(ds); d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "ls_params_heatmap.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        sc = ax.scatter(agg["t"], agg["HV"], c=agg["Q"], cmap="viridis", s=60)
        for _, r in agg.iterrows():
            ax.annotate(f"p={r.p_ls:g},I={r.i_ls:g}", (r.t, r.HV), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Thời gian trung bình (s)"); ax.set_ylabel("Hypervolume trung bình"); ax.set_title(f"Chất lượng nghiệm và chi phí – {ds}")
        fig.colorbar(sc, label="Q trung bình"); fig.tight_layout()
        fig.savefig(d / "ls_params_tradeoff.png", dpi=160); plt.close(fig)
        base = agg[agg["p_ls"] == 0]
        if not base.empty:
            b = base.iloc[0]
            tr = agg.copy()
            tr["ΔHV so với p_LS=0"] = (tr["HV"] - b["HV"]).map("{:+.4f}".format)
            tr["Thời gian gấp (lần)"] = (tr["t"] / b["t"]).map("{:.2f}".format)
            tr["ΔHV mỗi giây thêm"] = [f"{(h - b['HV']) / (tt - b['t']):.5f}" if tt > b["t"] else "–" for h, tt in zip(agg["HV"], agg["t"])]
            tr = tr[["p_ls", "i_ls", "ΔHV so với p_LS=0", "Thời gian gấp (lần)", "ΔHV mỗi giây thêm"]].rename(columns={"p_ls": "p_LS", "i_ls": "I_LS"})
            rep.table(f"t3_4_3_tradeoff_{safe(ds)}", f"3.4.3 Cân bằng chất lượng nghiệm và chi phí tính toán ({ds})", tr)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["results/main/runs.jsonl"])
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default="results/report")
    ap.add_argument("--datasets", nargs="*", default=None, help="chỉ vẽ các bộ dữ liệu này")
    a = ap.parse_args()
    out = Path(a.out)
    rep = Report(out)
    df = load_runs(a.runs)
    if (df["tag"] == "default").any():
        df = df[df["tag"] == "default"]
    if a.datasets:
        df = df[df["dataset"].isin(a.datasets)]
    profiles, graphs = load_profiles(a.runs)
    sec_datasets(profiles, df, rep); sec_32(df, rep); sec_33(df, rep); sec_34(df, rep); sec_per_dataset(df, rep)
    summary_long(df, out)
    n_fig = 0
    for ds, sub in df.groupby("dataset", observed=True):
        made = rp.per_dataset_figures(sub, str(ds), out / "figures", profiles.get(str(ds)), graphs.get(str(ds)))
        n_fig += len(made)
        print(f"  {ds}: {len(made)} hình")
    if df["dataset"].nunique() > 1:
        n_fig += len(rp.summary_figures(df, out / "figures"))
    if a.sweep and Path(a.sweep).exists():
        sec_34_sweep(load_runs(a.sweep), rep, out)
    rep.save()
    print(f"Đã ghi {out / 'tables.md'} và {n_fig} hình trong {out / 'figures'}")


if __name__ == "__main__":
    main()
