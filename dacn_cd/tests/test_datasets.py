"""Kiểm thử bộ nạp dữ liệu bằng tệp mô phỏng đúng định dạng. Chạy: python tests/test_datasets.py"""
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from DACN.dacn_cd.cdga import datasets
from DACN.dacn_cd.tests.fixtures import build                      # noqa: E402
from DACN.dacn_cd.cdga import metrics               # noqa: E402


def test_loaders_on_simulated_files():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DACN_DATA"] = str(build(Path(tmp)))
        fb = datasets.load("football")
        assert fb.n >= 55 and fb.truth.max() + 1 == 6 and (fb.truth >= 0).all()
        pb = datasets.load("polbooks")
        assert pb.stats["n_raw"] == 100 and pb.n >= 95 and pb.truth.max() + 1 == 3   # đỉnh cô lập bị loại là đúng

        bl = datasets.load("polblogs")
        assert bl.stats["directed_raw"] and bl.stats["n_raw"] == 123           # 120 + cô lập + 2 đỉnh phụ
        assert bl.n <= 120 and bl.deg.min() >= 1                               # chỉ giữ thành phần liên thông lớn nhất
        assert len(set(zip(bl.src.tolist(), bl.dst.tolist()))) == bl.m         # không có cạnh trùng / self-loop
        assert (bl.src != bl.dst).all()

        em = datasets.load("email")
        assert em.stats["m_raw_input"] > em.m and em.truth.max() + 1 == 6      # hai chiều gộp thành một cạnh

        am = datasets.load("amazon")
        assert (am.truth == -1).any() and (am.truth >= 0).any()                # đỉnh thuộc 0 hoặc >= 2 cộng đồng bị loại
        assert am.stats["n_truth_labeled"] < am.n
        pred = np.zeros(am.n, dtype=np.int32)
        v = metrics.nmi(am.truth, pred)                                        # không lỗi khi có nhãn -1
        assert 0.0 <= v <= 1.0
        del os.environ["DACN_DATA"]


def test_missing_file_message():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DACN_DATA"] = tmp
        try:
            datasets.load("email")
            raise AssertionError("phải báo thiếu tệp")
        except FileNotFoundError as e:
            assert "snap.stanford.edu" in str(e)
        finally:
            del os.environ["DACN_DATA"]


def test_duplicate_gml_edges_are_deduplicated():
    with tempfile.TemporaryDirectory() as tmp:
        d = build(Path(tmp))
        os.environ["DACN_DATA"] = str(d)
        try:
            path = d / "polblogs.gml"
            original = path.read_text(encoding="utf-8")
            baseline = datasets.load("polblogs")
            match = re.search(
                r"(?m)^\s*edge \[\s*source\s+(\d+)\s*target\s+(\d+)\s*\]",
                original,
            )
            assert match is not None
            assert original.rstrip().endswith("]")
            duplicate = (
                f"\n  edge [\n    source {match.group(1)}\n"
                f"    target {match.group(2)}\n  ]\n]"
            )
            path.write_text(original.rstrip()[:-1] + duplicate, encoding="utf-8")

            graph = datasets.load("polblogs")
            assert graph.stats["m_raw_input"] == baseline.stats["m_raw_input"] + 1
            assert graph.m == baseline.m
            assert graph.stats["directed_raw"]
        finally:
            del os.environ["DACN_DATA"]


def test_non_ascii_gml_amazon_top_and_profile():
    from DACN.dacn_cd.cdga.profile import dataset_profile
    with tempfile.TemporaryDirectory() as tmp:
        d = build(Path(tmp))
        t = (d / "polbooks.gml").read_text(encoding="utf-8").replace('Book 1"', 'Book 1 – Rumsfeld’s war"', 1)
        (d / "polbooks.gml").write_text(t, encoding="utf-8")                  # ký tự ngoài ASCII
        os.environ["DACN_DATA"] = str(d)
        pb = datasets.load("polbooks")
        assert pb.n >= 95
        top = datasets.load("amazon:top=3")
        assert top.name == "amazon_top3" and top.n < datasets.load("amazon").n
        assert top.truth.max() + 1 <= 3
        prof = dataset_profile(datasets.load("football"))
        assert prof["truth_K"] == 6 and sum(prof["degree_counts"]) == prof["n"] and 0 <= prof["mixing"] <= 1
        assert prof["truth_Q"] is not None and sum(prof["truth_sizes"]) == prof["n_labeled"]
        del os.environ["DACN_DATA"]


if __name__ == "__main__":
    test_loaders_on_simulated_files(); print("ok   test_loaders_on_simulated_files")
    test_missing_file_message(); print("ok   test_missing_file_message")
    test_duplicate_gml_edges_are_deduplicated(); print("ok   test_duplicate_gml_edges_are_deduplicated")
    test_non_ascii_gml_amazon_top_and_profile(); print("ok   test_non_ascii_gml_amazon_top_and_profile")
