"""Thước đo ngoại tại NMI, ARI (mục 2.5.4c). NMI chuẩn hóa theo trung bình cộng (sklearn).

Đỉnh không có nhãn ground-truth (truth = -1, ví dụ Amazon) bị loại khỏi phép tính.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _masked(truth: np.ndarray, pred: np.ndarray):
    m = truth >= 0
    return truth[m], pred[m]


def nmi(truth: np.ndarray, pred: np.ndarray) -> float:
    t, p = _masked(truth, pred)
    return float(normalized_mutual_info_score(t, p, average_method="arithmetic"))


def ari(truth: np.ndarray, pred: np.ndarray) -> float:
    t, p = _masked(truth, pred)
    return float(adjusted_rand_score(t, p))
