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
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions, split_summary
from thresholds import (
    best_threshold,
    brute_force_best,
    brute_force_table,
    confusion,
    threshold_table,
)

LINE = "#2a78d6"
MUTED = "#898781"

CHECK_SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}


def fmt_t(t: float) -> str:
    return "+inf" if np.isinf(t) else f"{t:.6g}"


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def plot_curves(y: np.ndarray, s: np.ndarray, t_star: float, auc: float, out: Path) -> None:
    marks = {"t = 0.5": 0.5, "t* = " + fmt_t(t_star): t_star}

    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=LINE, lw=2, drawstyle="steps-post")
    for (label, t), marker in zip(marks.items(), ("o", "s"), strict=True):
        m = confusion(y, s, t)
        p = ratio(m["TP"], m["TP"] + m["FP"])
        r = ratio(m["TP"], m["TP"] + m["FN"])
        if p is not None:
            ax.plot(r, p, marker, color="black", ms=8, label=label)
    ax.axhline(y.mean(), color=MUTED, lw=1, ls="--", label=f"частка позитивних = {y.mean():.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR-крива, validation", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color=LINE, lw=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for (label, t), marker in zip(marks.items(), ("o", "s"), strict=True):
        m = confusion(y, s, t)
        ax.plot(m["FP"] / (m["FP"] + m["TN"]), m["TP"] / (m["TP"] + m["FN"]), marker,
                color="black", ms=8, label=label)
    ax.set(xlabel="FPR", ylabel="TPR (recall)", title="ROC-крива, validation")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ins = ax.inset_axes([0.38, 0.4, 0.4, 0.33])
    ins.plot(fpr, tpr, color=LINE, lw=2)
    for t, marker in zip(marks.values(), ("o", "s"), strict=True):
        m = confusion(y, s, t)
        ins.plot(m["FP"] / (m["FP"] + m["TN"]), m["TP"] / (m["TP"] + m["FN"]), marker, color="black", ms=7)
    ins.set_xlim(0, 0.01)
    ins.set_ylim(0.5, 1.0)
    ins.set_title("FPR до 0.01", fontsize=9)
    ins.tick_params(labelsize=8)
    ins.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)


def run_checks(out: Path) -> bool:
    all_ok = True
    rows = []
    for name, (s, y) in CHECK_SETS.items():
        fast = threshold_table(np.array(y), np.array(s))
        slow = brute_force_table(y, s)
        same_table = fast.equals(slow.astype(fast.dtypes.to_dict()))
        t_fast = float(best_threshold(fast)["threshold"])
        t_slow = brute_force_best(slow)
        ok = same_table and t_fast == t_slow
        all_ok &= ok

        print(f"\nНабір {name}: s = {s}, y = {y}")
        shown = slow.copy()
        shown["threshold"] = shown["threshold"].map(fmt_t)
        print(shown.to_string(index=False))
        print(f"  t* (перебір) = {fmt_t(t_slow)}, t* (кумулятивні суми) = {fmt_t(t_fast)}, "
              f"таблиці збігаються: {same_table} -> {'пройдено' if ok else 'НЕ ПРОЙДЕНО'}")
        slow.to_csv(out / f"check_{name}.csv", index=False)
        rows.append({"set": name, "t_star_brute": t_slow, "t_star_fast": t_fast,
                     "tables_equal": same_table, "passed": ok})
    pd.DataFrame(rows).to_csv(out / "checks_summary.csv", index=False)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(args.saved)
    print("Поділ:")
    print(split_summary(data).to_string(index=False))

    y_va, s_va = data["validation_y"], data["validation_scores"]
    auc = float(roc_auc_score(y_va, s_va))
    table = threshold_table(y_va, s_va)
    table.to_csv(out / "validation_thresholds.csv", index=False, float_format="%.17g")
    best = best_threshold(table)
    t_star = float(best["threshold"])

    summary = pd.DataFrame(
        [
            {"rule": "Стандартний поріг", "threshold": 0.5, **confusion(y_va, s_va, 0.5)},
            {"rule": "Мінімум вартості", "threshold": t_star, **confusion(y_va, s_va, t_star)},
            {"rule": "Усі негативні", "threshold": np.inf, **confusion(y_va, s_va, np.inf)},
        ]
    )
    assert (summary.iloc[1][["TP", "FP", "FN", "TN", "C"]].to_numpy()
            == best[["TP", "FP", "FN", "TN", "C"]].to_numpy()).all()
    summary.to_csv(out / "validation_summary.csv", index=False, float_format="%.17g")
    n_min = int((table["C"] == table["C"].min()).sum())

    print(f"\nValidation: ROC-AUC = {auc:.6f}, кандидатів = {len(table)}, "
          f"порогів з мінімальною C = {n_min}")
    shown = summary.copy()
    shown["threshold"] = shown["threshold"].map(fmt_t)
    print(shown.to_string(index=False))
    print(f"t* = {t_star!r}")
    plot_curves(y_va, s_va, t_star, auc, out)

    print("\nПеревірка пошуку порога на малих наборах")
    if not run_checks(out):
        print("\nПеревірки не пройдено, тестова оцінка не виконується.")
        sys.exit(1)

    y_te, s_te, a_te, rid_te = (data[f"test_{k}"] for k in ("y", "scores", "amount", "row_ids"))
    m = confusion(y_te, s_te, t_star)
    precision = ratio(m["TP"], m["TP"] + m["FP"])
    recall = ratio(m["TP"], m["TP"] + m["FN"])
    print(f"\nTest при t* = {t_star!r}: {m}")
    print(f"precision = {precision}, recall = {recall}")

    pred = s_te >= t_star
    err = pd.DataFrame({"row": rid_te, "label": y_te, "score": s_te.astype(np.float64), "amount": a_te})
    err["type"] = np.where(pred & (y_te == 0), "FP", np.where(~pred & (y_te == 1), "FN", ""))
    err["score_minus_t"] = err["score"] - t_star
    examples = pd.concat(
        [err[err["type"] == k].sort_values("row").head(5) for k in ("FP", "FN")]
    )[["row", "type", "label", "score", "amount", "score_minus_t"]]
    examples.to_csv(out / "test_errors.csv", index=False, float_format="%.17g")
    print("\nПерші п'ять FP і FN на test (за рядком CSV):")
    print(examples.to_string(index=False))

    result = {
        "t_star": t_star,
        "validation_roc_auc": auc,
        "validation": summary.assign(threshold=summary["threshold"].map(fmt_t)).to_dict("records"),
        "test": {**m, "precision": precision, "recall": recall},
    }
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nРезультати збережено у {out.resolve()}")


if __name__ == "__main__":
    main()
