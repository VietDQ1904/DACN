"""Chạy thực nghiệm (đề cương Chương 3, mục 3.2 - 3.4).

Ví dụ:
  python scripts/run_experiments.py --datasets karate polbooks football polblogs email \\
      --algos GA HGA NSGA2 MO-HGA-LS Louvain Leiden --runs 30 --out results/main --jobs 4

  # Amazon (334k đỉnh): chỉ chạy được Louvain/Leiden; GA bị bỏ qua tự động (--max_ga_nodes)
  python scripts/run_experiments.py --datasets amazon --algos Louvain Leiden --runs 5 --out results/amazon

  # khảo sát tham số Local Search (mục 3.4.2)
  python scripts/run_experiments.py --datasets football --algos MO-HGA-LS --runs 10 \\
      --grid "p_ls=0,0.1,0.2,0.5;i_ls=1,5,10,30" --out results/sweep_ls

  # cấu hình nhanh cho đồ thị lớn (Email): --preset fast
Mỗi lần chạy là một dòng JSON trong <out>/runs.jsonl (tự bỏ qua các lần đã chạy -> dừng/chạy tiếp được).
Cùng thư mục còn có datasets.json (thống kê từng bộ dữ liệu) và graphs/*.npz (cạnh + nhãn chuẩn của đồ thị nhỏ,
dùng để vẽ đồ thị phân hoạch); scripts/analyze.py tự đọc các tệp này.
Lưu ý: chạy song song (--jobs > 1) làm thời gian đo được nhiễu hơn; muốn đo thời gian chính xác dùng --jobs 1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdga import ALGOS, Config, Engine, baselines, load, metrics, select_solutions   # noqa: E402
from cdga import pareto                      # noqa: E402
from cdga.profile import dataset_profile, safe, save_graph_npz   # noqa: E402

BASELINES = ("Louvain", "Leiden")
PRESETS = {
    "paper": {},   # đúng cấu hình mô tả trong báo cáo (µ=100, G_max=200, I_LS=30)
    "fast": dict(mu=50, lam=50, g_max=100, k_stag=20, i_ls=5, p_ls=0.1),
}
_CACHE: dict = {}


def get_graph(name):
    if name not in _CACHE:
        _CACHE[name] = load(name)
    return _CACHE[name]


def parse_value(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def expand_grid(grid):
    """'p_ls=0,0.1;i_ls=1,5' -> [{'p_ls':0,'i_ls':1}, ...]"""
    if not grid:
        return [{}]
    keys, vals = [], []
    for part in grid.split(";"):
        k, v = part.split("=")
        keys.append(k.strip())
        vals.append([parse_value(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def tag_of(over: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(over.items())) or "default"


def sol_metrics(g, labels, q, phi):
    d = {"Q": float(q), "Phi": float(phi), "K": int(labels.max()) + 1}
    if g.truth is not None and (g.truth >= 0).any():
        d["NMI"] = metrics.nmi(g.truth, labels)
        d["ARI"] = metrics.ari(g.truth, labels)
    return d


def truth_info(g) -> dict:
    """Q, Φ, K của phân hoạch chuẩn (chỉ khi mọi đỉnh đều có nhãn và K >= 2)."""
    out = {"truth_Q": None, "truth_Phi": None, "truth_K": None}
    if g.truth is not None:
        out["truth_K"] = int(g.truth.max()) + 1
        if (g.truth >= 0).all() and g.truth.max() >= 1:
            q, phi = g.objectives(g.truth.astype(np.int32))
            out["truth_Q"], out["truth_Phi"] = float(q), float(phi)
    return out


def run_one(task):
    dataset, algo, seed, base_cfg, over, opts = task
    g = get_graph(dataset)
    keep_labels = g.n <= opts["labels_max_n"]
    row = {"dataset": dataset, "algo": algo, "seed": seed, "tag": tag_of(over),
           "n": g.n, "m": g.m, **truth_info(g)}
    if algo in BASELINES:
        t0 = time.perf_counter()
        if algo == "Louvain":
            labels, impl = baselines.louvain(g, seed=seed), "networkx"
        else:
            labels, impl = baselines.leiden(g, seed=seed)
        row["time_s"] = time.perf_counter() - t0
        row["impl"] = impl
        q, phi = g.objectives(labels)
        row["solutions"] = {"maxQ": sol_metrics(g, labels, q, phi)}
        F = pareto.to_min([q], [phi], True)
        row.update(hv_final=pareto.hypervolume(F), front=[[q, phi]], generations=None, n_eval=None,
                   n_ls_eval=None, time_ls=None, archive_size=1, hv_history=[], q_history=[],
                   archive_size_history=[], time_history=[],
                   labels={"maxQ": labels.tolist()} if keep_labels else None)
        return row
    cfg = Config(**{**base_cfg, **over})
    eng = Engine(g, algo, cfg, seed=seed)
    archive = eng.run()
    sols = select_solutions(archive, eng.multi)
    row["time_s"] = eng.time
    row["solutions"] = {k: sol_metrics(g, s.labels, s.q, s.phi) for k, s in sols.items()}
    F = pareto.to_min([s.q for s in archive], [s.phi for s in archive], True)
    row.update(hv_final=pareto.hypervolume(F),
               front=sorted([[s.q, s.phi] for s in archive]),
               generations=eng.generations, n_eval=eng.n_eval, n_ls_eval=eng.ls.eval_count,
               time_ls=eng.t_ls, archive_size=len(archive), hv_history=eng.hv_history,
               q_history=eng.q_history, archive_size_history=eng.archive_size_history,
               time_history=eng.t_history, ls_calls=eng.ls_calls, ls_changed=eng.ls_changed,
               ls_improved=eng.ls_improved, ls_dq=eng.ls_dq, ls_dphi=eng.ls_dphi,
               labels={k: s.labels.tolist() for k, s in sols.items()} if keep_labels else None,
               config=cfg.to_dict())
    return row


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS) + list(BASELINES))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--grid", default=None, help='vd: "p_ls=0,0.2,0.5;i_ls=5,30" (áp dụng cho GA/HGA/NSGA2/MO-HGA-LS)')
    ap.add_argument("--preset", choices=list(PRESETS), default="paper")
    ap.add_argument("--max_ga_nodes", type=int, default=20000,
                    help="bỏ qua các thuật toán GA nếu đồ thị có nhiều đỉnh hơn ngưỡng này")
    ap.add_argument("--save_labels_max_n", type=int, default=3000,
                    help="lưu nhãn phân hoạch từng lần chạy và cạnh đồ thị nếu n <= ngưỡng (để vẽ đồ thị phân hoạch)")
    for f in fields(Config):
        ap.add_argument(f"--{f.name}", type=parse_value, default=None)
    args = ap.parse_args()

    if "Leiden" in args.algos and not baselines.leiden_available():
        print("CẢNH BÁO: thiếu igraph/leidenalg -> bỏ qua Leiden (pip install igraph leidenalg).", flush=True)
        args.algos = [a for a in args.algos if a != "Leiden"]

    base_cfg = dict(PRESETS[args.preset])
    base_cfg.update({f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["dataset"], r["algo"], r["seed"], r["tag"]))

    prof_path = out / "datasets.json"
    profiles = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else {}
    opts = {"labels_max_n": args.save_labels_max_n}
    tasks = []
    for ds in args.datasets:
        try:
            g = get_graph(ds)
        except FileNotFoundError as e:
            print(f"BỎ QUA {ds}: {e}", flush=True)
            continue
        print(f"{ds}: n={g.n}, m={g.m} (sau tiền xử lý), <k>={2 * g.m / g.n:.2f}, "
              f"nhãn chuẩn: {g.stats.get('n_truth_labeled')}/{g.n} đỉnh", flush=True)
        profiles[ds] = dataset_profile(g)
        if g.n <= args.save_labels_max_n:
            save_graph_npz(g, out / "graphs" / f"{safe(ds)}.npz")
        prof_path.write_text(json.dumps(profiles), encoding="utf-8")
        for algo in args.algos:
            if algo in ALGOS and g.n > args.max_ga_nodes:
                print(f"  bỏ qua {algo} trên {ds}: n={g.n} > --max_ga_nodes={args.max_ga_nodes}", flush=True)
                continue
            for over in (expand_grid(args.grid) if algo in ALGOS else [{}]):
                for r in range(args.runs):
                    seed = args.seed0 + r
                    if (ds, algo, seed, tag_of(over)) not in done:
                        tasks.append((ds, algo, seed, base_cfg, over, opts))
    print(f"{len(tasks)} lần chạy cần thực hiện ({len(done)} đã có) -> {path}", flush=True)
    (out / "config.json").write_text(json.dumps({"base_cfg": base_cfg, "args": vars(args)}, indent=1, default=str),
                                     encoding="utf-8")

    t0 = time.time()
    with path.open("a", encoding="utf-8") as fh:
        if args.jobs > 1:
            with mp.Pool(args.jobs) as pool:
                for i, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    _progress(i, len(tasks), row, t0)
        else:
            for i, task in enumerate(tasks, 1):
                row = run_one(task)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                _progress(i, len(tasks), row, t0)


def _progress(i, total, row, t0):
    q = row["solutions"]["maxQ"]["Q"]
    print(f"[{i}/{total}] {row['dataset']:<14} {row['algo']:<10} seed={row['seed']:<3} "
          f"{row['tag']:<18} Q={q:.4f} t={row['time_s']:.1f}s  (tổng {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
