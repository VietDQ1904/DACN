"""Tải các bộ dữ liệu vào data/ (chạy trên máy có Internet). Tệp đã có sẽ được bỏ qua.

  python scripts/download_datasets.py                 # polbooks, football, polblogs, email-Eu-core
  python scripts/download_datasets.py --amazon        # thêm com-amazon (ungraph + nhãn cộng đồng)

Nếu một địa chỉ lỗi, tải tay theo README.md rồi đặt tệp vào data/ (tên tệp: xem cdga/datasets.py).
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SNAP = "https://snap.stanford.edu/data/"
NEWMAN = ["https://public.websites.umich.edu/~mejn/netdata/", "https://websites.umich.edu/~mejn/netdata/"]


def fetch(url):
    print("tải", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def try_fetch(urls):
    last = None
    for u in urls:
        try:
            return fetch(u)
        except Exception as e:   # noqa: BLE001
            last = e
            print("  lỗi:", e, flush=True)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amazon", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for name in ("polbooks", "football", "polblogs"):
        if (DATA / f"{name}.gml").exists():
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(try_fetch([u + f"{name}.zip" for u in NEWMAN])))
            gml = next(n for n in z.namelist() if n.lower().endswith(".gml"))
            (DATA / f"{name}.gml").write_bytes(z.read(gml))
        except Exception as e:   # noqa: BLE001
            print(f"KHÔNG TẢI ĐƯỢC {name}: {e}\n  -> tải tay tại {NEWMAN[0]}", flush=True)
    files = ["email-Eu-core.txt.gz", "email-Eu-core-department-labels.txt.gz"]
    if a.amazon:
        files += ["bigdata/communities/com-amazon.ungraph.txt.gz", "bigdata/communities/com-amazon.all.dedup.cmty.txt.gz"]
    for f in files:
        dst = DATA / Path(f).name
        if dst.exists() or (DATA / dst.stem).exists():
            continue
        try:
            dst.write_bytes(fetch(SNAP + f))
        except Exception as e:   # noqa: BLE001
            print(f"KHÔNG TẢI ĐƯỢC {f}: {e}", flush=True)
    print("xong ->", DATA)


if __name__ == "__main__":
    main()
