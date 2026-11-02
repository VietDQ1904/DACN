"""Kiểm thử ghép các lần chạy khi tổng hợp kết quả."""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DACN.dacn_cd.scripts.analyze import Report, load_runs, sec_33  # noqa: E402


def test_wilcoxon_pairs_match_seed_and_config():
    with tempfile.TemporaryDirectory() as tmp:
        runs_path = Path(tmp) / "runs.jsonl"
        with runs_path.open("w", encoding="utf-8") as fh:
            for algo, config in (
                ("GA", {"g_max": 200}),
                ("GA", {"g_max": 100}),
                ("MO-HGA-LS", {"g_max": 100}),
            ):
                for seed in range(6):
                    run = {
                        "dataset": "polblogs",
                        "algo": algo,
                        "seed": seed,
                        "tag": "default",
                        "config": config,
                        "solutions": {
                            "maxQ": {"Q": 0.5, "Phi": 0.2, "K": 3, "NMI": 0.8, "ARI": 0.7},
                            "knee": {"Q": 0.4, "Phi": 0.2, "K": 3, "NMI": 0.7, "ARI": 0.6},
                            "minPhi": {"Q": 0.3, "Phi": 0.1, "K": 4, "NMI": 0.6, "ARI": 0.5},
                        },
                        "hv_final": 0.6,
                        "time_s": 1.0,
                        "time_ls": 0.1,
                        "generations": 10,
                        "n_eval": 20,
                        "n_ls_eval": 5,
                    }
                    fh.write(json.dumps(run) + "\n")

        df = load_runs(runs_path)
        report = Report(Path(tmp) / "report")
        sec_33(df, report)

        table = Path(tmp) / "report" / "tables" / "t3_3_4b_wilcoxon.csv"
        with table.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        assert {row["n cặp"] for row in rows} == {"6"}


if __name__ == "__main__":
    test_wilcoxon_pairs_match_seed_and_config()
    print("ok   test_wilcoxon_pairs_match_seed_and_config")
