"""Kiểm tra thư mục data/: tệp nào đã có, bộ dữ liệu nào nạp được và thống kê sau tiền xử lý.

  python scripts/check_datasets.py                       # karate polbooks football polblogs email
  python scripts/check_datasets.py --datasets amazon amazon:top=20
Đổi thư mục dữ liệu: đặt biến môi trường DACN_DATA.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdga import datasets   # noqa: E402

NEED = {
    "karate": [],
    "polbooks": ["polbooks.gml"],
    "football": ["football.gml"],
    "polblogs": ["polblogs.gml"],
    "email": ["email-Eu-core.txt", "email-Eu-core-department-labels.txt"],
    "amazon": ["com-amazon.ungraph.txt", "com-amazon.all.dedup.cmty.txt"],
}


def have(files, name):
    return name.lower() in files or (name + ".gz").lower() in files


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["karate", "polbooks", "football", "polblogs", "email"])
    a = ap.parse_args()
    d = datasets.data_dir()
    files = {p.name.lower() for p in d.iterdir()} if d.exists() else set()
    print(f"Thư mục dữ liệu: {d}  ({'tồn tại' if d.exists() else 'KHÔNG tồn tại'}, {len(files)} tệp)\n")
    ok_all = True
    for ds in a.datasets:
        base = ds.split(":")[0].lower()
        miss = [f for f in NEED.get(base, []) if not have(files, f)]
        if base == "amazon" and miss == ["com-amazon.all.dedup.cmty.txt"] and have(files, "com-amazon.top5000.cmty.txt"):
            miss = []
        if miss:
            ok_all = False
            print(f"[THIẾU] {ds}: không thấy {', '.join(miss)}")
            hint = datasets.SNAP_HINT.get("email" if base == "email" else "amazon" if base == "amazon" else "newman")
            print(f"        tải tại: {hint}\n")
            continue
        t0 = time.time()
        try:
            g = datasets.load(ds)
        except Exception as e:   # noqa: BLE001
            ok_all = False
            print(f"[LỖI]   {ds}: {type(e).__name__}: {e}\n")
            continue
        s = g.stats
        print(f"[OK]    {ds}  ({time.time() - t0:.1f}s)")
        print(f"        đầu vào: {s.get('n_raw')} đỉnh, {s.get('m_raw_input')} cạnh"
              f"{' (có hướng)' if s.get('directed_raw') else ''}")
        print(f"        sau tiền xử lý: n={g.n}, m={g.m}, bậc TB={s['avg_degree']:.2f}, CC TB={s['avg_cc']:.4f}")
        if g.truth is not None:
            print(f"        nhãn chuẩn: {s['n_truth_labeled']}/{g.n} đỉnh, {s['n_truth_communities']} cộng đồng")
        else:
            print("        nhãn chuẩn: không có (NMI/ARI sẽ bỏ trống)")
        print()
    print("Mọi bộ dữ liệu đều sẵn sàng." if ok_all else "Có bộ dữ liệu thiếu hoặc lỗi (xem ở trên).")


if __name__ == "__main__":
    main()
