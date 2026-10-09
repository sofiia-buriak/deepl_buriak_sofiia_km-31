from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def cost(fn: int | np.ndarray, fp: int | np.ndarray) -> int | np.ndarray:
    return COST_FN * fn + COST_FP * fp


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    y = np.asarray(y)
    pred = np.asarray(s) >= t
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": int(cost(fn, fp))}


def threshold_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    last = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])

    thr = np.r_[np.inf, s_sorted[last].astype(np.float64)]
    tp = np.r_[0, tp_cum[last]]
    fp = np.r_[0, fp_cum[last]]
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(np.count_nonzero(y == 0))
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}
    )


def best_threshold(table: pd.DataFrame) -> pd.Series:
    at_min = table[table["C"] == table["C"].min()]
    return at_min.loc[at_min["threshold"].idxmax()]


def brute_force_table(y: list | np.ndarray, s: list | np.ndarray) -> pd.DataFrame:
    ys = [int(v) for v in y]
    ss = [float(v) for v in s]
    candidates = [float("inf")] + sorted(set(ss), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for yi, si in zip(ys, ss, strict=True):
            if si >= t:
                if yi == 1:
                    tp += 1
                else:
                    fp += 1
            elif yi == 1:
                fn += 1
            else:
                tn += 1
        rows.append([t, tp, fp, fn, tn, 10 * fn + fp])
    return pd.DataFrame(rows, columns=COLUMNS)


def brute_force_best(rows: pd.DataFrame) -> float:
    best_t, best_c = None, None
    for t, c in zip(rows["threshold"], rows["C"], strict=True):
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_t, best_c = t, c
    return float(best_t)
