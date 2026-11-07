from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix

from common import (
    CLASSES, FIGURES, MODEL_FILE, RUN_PLAN, RUNS, SPLIT_CSV, SplitDataset, eval_loader,
    eval_transform, logits_of, macro_f1, read_split, sha256,
)

OUT = Path(__file__).resolve().parent / "analysis"
COLORS = {"scratch": "#c0392b", "frozen": "#2a78d6", "unfreeze": "#1e8f4e"}
STYLES = {1: "-", 2: "--", 3: ":"}


def p(text: str = "", out: list[str] | None = None) -> None:
    print(text, flush=True)
    if out is not None:
        out.append(text)


def load_runs() -> dict[tuple[str, int], dict]:
    runs = {}
    for regime, seed in RUN_PLAN:
        f = RUNS / f"{regime}_seed{seed}" / "run.json"
        if not f.exists():
            sys.exit(f"Немає журналу {f}")
        runs[(regime, seed)] = json.loads(f.read_text(encoding="utf-8"))
    return runs


def check_hash(runs: dict, rep: list[str]) -> None:
    current = sha256(SPLIT_CSV)
    hashes = {r["split_csv_sha256"] for r in runs.values()}
    ok = hashes == {current}
    p(f"SHA-256 поточного split.csv: {current}", rep)
    p(f"Хеш однаковий у всіх семи журналах і збігається з поточним: {'так' if ok else 'НІ'}", rep)
    if not ok:
        sys.exit("Запуски зроблено на іншому поділі: перенавчіть усі сім.")
    commits = {(r["git"]["commit"], r["git"]["uncommitted_changes"]) for r in runs.values()}
    p(f"Коміти коду в журналах: {sorted(commits, key=str)}", rep)


def runs_table(runs: dict, rep: list[str]) -> None:
    p("", rep)
    p("## Сім запусків", rep)
    p("", rep)
    p("| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |", rep)
    p("|---|---|---|---|---|---|", rep)
    for i, ((regime, seed), r) in enumerate(runs.items(), 1):
        p(f"| {i} | {regime} | {seed} | {r['final_train_loss']:.4f} | {r['final_val_macro_f1']:.4f} "
          f"| {r['train_time_s']:.0f} |", rep)
    devices = {r["device"] for r in runs.values()}
    p("", rep)
    p(f"Пристрій навчання: {', '.join(sorted(devices))}", rep)


def plot_curves(runs: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (regime, seed), r in runs.items():
        ep = [h["epoch"] for h in r["history"]]
        kw = dict(color=COLORS[regime], ls=STYLES[seed], lw=1.8, marker="o", ms=3,
                  label=f"{regime}, зерно {seed}")
        a.plot(ep, [h["train_loss"] for h in r["history"]], **kw)
        b.plot(ep, [h["val_macro_f1"] for h in r["history"]], **kw)
    a.set(xlabel="Епоха", ylabel="Втрата на train", title="Втрата на train")
    b.set(xlabel="Епоха", ylabel="macro-F1 на validation", title="macro-F1 на validation", ylim=(0, 1))
    for ax in (a, b):
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.3)
    b.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "curves.png", dpi=150)
    plt.close(fig)


def effects(runs: dict, rep: list[str]) -> dict:
    F = {k: r["final_val_macro_f1"] for k, r in runs.items()}
    mean = {g: float(np.mean([F[(g, s)] for s in (1, 2, 3)])) for g in ("frozen", "unfreeze")}
    rng = {g: max(F[(g, s)] for s in (1, 2, 3)) - min(F[(g, s)] for s in (1, 2, 3)) for g in mean}
    R = max(rng.values())
    d_pre = mean["frozen"] - F[("scratch", 1)]
    d_unf = mean["unfreeze"] - mean["frozen"]

    def verdict(d: float) -> str:
        return ("|Δ| > R: ефект вартий перевірки, але не доведений" if abs(d) > R
                else "|Δ| ≤ R: різниця не відрізняється від розкиду")

    p("", rep)
    p("## Розклад ефектів", rep)
    p("", rep)
    p("| Величина | Значення |", rep)
    p("|---|---|", rep)
    p(f"| F̄_frozen, R_frozen | {mean['frozen']:.4f}, {rng['frozen']:.4f} |", rep)
    p(f"| F̄_unfreeze, R_unfreeze | {mean['unfreeze']:.4f}, {rng['unfreeze']:.4f} |", rep)
    p(f"| F_scratch,1 | {F[('scratch', 1)]:.4f} |", rep)
    p(f"| R | {R:.4f} |", rep)
    p(f"| Δpre = F̄_frozen − F_scratch,1 | {d_pre:+.4f} |", rep)
    p(f"| Δunf = F̄_unfreeze − F̄_frozen | {d_unf:+.4f} |", rep)
    p(f"| Δpre + Δunf = F̄_unfreeze − F_scratch,1 | {d_pre + d_unf:+.4f} |", rep)
    p("", rep)
    p(f"Δpre: {verdict(d_pre)} ({abs(d_pre):.4f} проти R = {R:.4f})", rep)
    p(f"Δunf: {verdict(d_unf)} ({abs(d_unf):.4f} проти R = {R:.4f})", rep)
    times = {g: float(np.mean([r["train_time_s"] for (gg, _), r in runs.items() if gg == g]))
             for g in ("scratch", "frozen", "unfreeze")}
    p("Середній час навчання, с: " + ", ".join(f"{g} {t:.0f}" for g, t in times.items()), rep)
    return {"F": F, "mean": mean, "range": rng, "R": R, "d_pre": d_pre, "d_unf": d_unf}


def expected_choice(runs: dict) -> tuple[str, int, float, dict]:
    means = {}
    for (g, _), r in runs.items():
        means.setdefault(g, []).append(r["final_val_macro_f1"])
    means = {g: float(np.mean(v)) for g, v in means.items()}
    regime = max(means, key=means.get)
    seed, f = max(((s, r["final_val_macro_f1"]) for (g, s), r in runs.items() if g == regime),
                  key=lambda t: t[1])
    return regime, seed, f, means


def check_choice(runs: dict, eff: dict, rows, data: Path, rep: list[str]):
    regime, seed, f_sel, means = expected_choice(runs)
    ckpt = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    p("", rep)
    p("## Вибір конфігурації", rep)
    p("", rep)
    p("Середні macro-F1 на validation: " + ", ".join(f"{g} {m:.4f}" for g, m in means.items()), rep)
    p(f"За правилом: режим {regime}, зерно {seed}, F = {f_sel:.4f}", rep)
    p(f"У lab3_model.pt: режим {ckpt['regime']}, зерно {ckpt['seed']}, img_size {ckpt['img_size']}, "
      f"класи {ckpt['classes']}", rep)
    ok = (ckpt["regime"], ckpt["seed"]) == (regime, seed) and ckpt["classes"] == CLASSES and ckpt["img_size"] == 160
    p(f"lab3_model.pt відповідає правилу вибору: {'так' if ok else 'НІ'}", rep)
    if not ok:
        sys.exit("lab3_model.pt не відповідає правилу вибору.")
    if regime in ("frozen", "unfreeze") and abs(eff["d_unf"]) <= eff["R"]:
        p("Увага: |Δunf| ≤ R, тож вибір між frozen і unfreeze визначив шум.", rep)

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_ds = SplitDataset(data, rows, "validation", eval_transform())
    y_val = np.array(val_ds.labels)
    z_val = logits_of(model, eval_loader(val_ds), torch.device("cpu"))
    saved = np.load(RUNS / f"{regime}_seed{seed}" / "val_logits.npy")
    agree = float((z_val.argmax(1) == saved.argmax(1)).mean())
    p(f"Звірка ваг: macro-F1 validation на CPU {macro_f1(y_val, z_val):.4f} (у журналі {f_sel:.4f}), "
      f"max |Δ логітів| {np.abs(z_val - saved).max():.2e}, збіг передбачень {agree:.4f}", rep)
    return model, regime, seed, f_sel


def test_eval(model, rows, data: Path, f_sel: float, rep: list[str]) -> None:
    test_ds = SplitDataset(data, rows, "test", eval_transform())
    y = np.array(test_ds.labels)
    z = logits_of(model, eval_loader(test_ds), torch.device("cpu"))
    np.save(OUT / "test_logits.npy", z)
    pred = z.argmax(1)
    f1 = macro_f1(y, z)
    acc = float((pred == y).mean())
    se = math.sqrt(acc * (1 - acc) / len(y))
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))

    p("", rep)
    p("## Тестова оцінка (CPU, один раз)", rep)
    p("", rep)
    p(f"macro-F1 на test = {f1:.4f}; F вибраного запуску на validation = {f_sel:.4f}; "
      f"різниця test − validation = {f1 - f_sel:+.4f}", rep)
    p(f"accuracy на test = {acc:.4f}; стандартна похибка √(p(1−p)/n) = {se:.4f} (n = {len(y)})", rep)
    p("", rep)
    p("Матриця плутанини (рядки - справжній клас, стовпці - передбачений):", rep)
    p("", rep)
    p("| справжній \\ передбачений | " + " | ".join(CLASSES) + " |", rep)
    p("|---" * (len(CLASSES) + 1) + "|", rep)
    for i, c in enumerate(CLASSES):
        p(f"| **{c}** | " + " | ".join(str(v) for v in cm[i]) + " |", rep)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(6), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас",
           title=f"Матриця плутанини, test (macro-F1 = {f1:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    files = {(i, j): sorted(test_ds.rows[k]["file"] for k in np.flatnonzero((y == i) & (pred == j)))
             for _, i, j in pairs}
    chosen, r = [], 0
    while len(chosen) < 5 and any(len(v) > r for v in files.values()):
        for _, i, j in pairs:
            if len(chosen) < 5 and len(files[(i, j)]) > r:
                chosen.append((files[(i, j)][r], i, j))
        r += 1
    prob = torch.softmax(torch.from_numpy(z), 1).numpy()
    index = {row["file"]: k for k, row in enumerate(test_ds.rows)}
    p("", rep)
    p("П'ять характерних помилок:", rep)
    p("", rep)
    p("| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test |", rep)
    p("|---|---|---|---|---|", rep)
    errors = []
    for f, i, j in chosen:
        k = index[f]
        errors.append({"file": f, "true": CLASSES[i], "pred": CLASSES[j], "prob": float(prob[k, j]),
                       "count": int(cm[i, j])})
        p(f"| {f} | {CLASSES[i]} | {CLASSES[j]} | {prob[k, j]:.4f} | {cm[i, j]} |", rep)

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.2 * len(chosen), 4.0), layout="constrained")
    for ax, e, (f, _, _) in zip(np.atleast_1d(axes), errors, chosen):
        ax.imshow(test_ds.image(index[f]))
        ax.set_title(f"{f}\nсправжній: {e['true']}\nмодель: {e['pred']} ({e['prob']:.2f})", fontsize=9)
        ax.axis("off")
    fig.savefig(FIGURES / "test_errors.png", dpi=150)
    plt.close(fig)

    result = {"test_macro_f1": f1, "test_accuracy": acc, "test_accuracy_se": se,
              "selected_val_macro_f1": f_sel, "confusion_matrix": cm.tolist(), "errors": errors}
    (OUT / "test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека, де лежать seg_train/ і seg_test/")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rep: list[str] = []
    runs = load_runs()
    check_hash(runs, rep)
    runs_table(runs, rep)
    plot_curves(runs)
    p("Криві: figures/curves.png", rep)
    eff = effects(runs, rep)
    rows = read_split()
    model, regime, seed, f_sel = check_choice(runs, eff, rows, args.data, rep)
    test_eval(model, rows, args.data, f_sel, rep)
    (OUT / "analysis.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\nЗвіт аналізу: analysis/analysis.md")


if __name__ == "__main__":
    main()
