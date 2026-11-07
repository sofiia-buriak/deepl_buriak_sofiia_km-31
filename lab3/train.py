from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from common import (
    CHECKS, CLASSES, EPOCHS, EXPECTED_SPLIT, EXPECTED_TOTAL, FIGURES, MEAN, MODEL_FILE, PARTS,
    REGIMES, RUN_PLAN, RUNS, SPLIT_CSV, STD, TOP_MODULES, SplitDataset, build_model, count,
    device_name, eval_loader, eval_transform, git_state, logits_of, macro_f1, make_split,
    pick_device, read_split, set_train_mode, settings, sha256, split_control, train_loader,
    train_transform, write_split,
)

CE = torch.nn.CrossEntropyLoss()


def log(text: str = "", out: list[str] | None = None) -> None:
    print(text, flush=True)
    if out is not None:
        out.append(text)

def check_split(data: Path, report: list[str]) -> list[dict[str, str]]:
    rows = make_split(data)
    write_split(rows)
    rows = read_split()
    table = split_control(rows)
    log("## Контроль поділу (кількість / сума імен)", report)
    log("", report)
    log("| Клас | train | validation | test | Збіг |", report)
    log("|---|---|---|---|---|", report)
    all_ok = True
    totals = {p: [0, 0] for p in PARTS}
    for c in CLASSES:
        cells, ok = [], True
        for p in PARTS:
            n, s = table[c][p]
            totals[p][0] += n
            totals[p][1] += s
            cells.append(f"{n} / {s:,}".replace(",", " "))
            ok &= (n, s) == EXPECTED_SPLIT[c][p]
        all_ok &= ok
        log(f"| {c} | " + " | ".join(cells) + f" | {'так' if ok else 'НІ'} |", report)
    log("| Разом | " + " | ".join(f"{n} / {s:,}".replace(",", " ") for n, s in totals.values()) + " | |", report)
    keys = [(r["label"], r["file"]) for r in rows]
    files = [r["file"] for r in rows]
    disjoint = len(set(keys)) == len(keys) and len(set(files)) == len(files)
    log("", report)
    log(f"Усі 18 клітинок збігаються з контрольною таблицею: {'так' if all_ok else 'НІ'}", report)
    log(f"Жоден файл не входить у дві частини: {'так' if disjoint else 'НІ'}", report)
    log(f"SHA-256 split.csv: {sha256(SPLIT_CSV)}", report)
    if not (all_ok and disjoint):
        sys.exit("Поділ не збігається з контрольною таблицею - далі не йдемо.")
    return rows


def save_batch_figure(ds: SplitDataset, report: list[str]) -> None:
    torch.manual_seed(0)
    x, y = next(iter(train_loader(ds)))
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.6))
    for ax, img, lab in zip(axes.flat, x[:16], y[:16]):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Навчальний пакет після аугментацій (нормування обернено), 160×160")
    fig.tight_layout()
    fig.savefig(FIGURES / "train_batch.png", dpi=120)
    plt.close(fig)
    log("", report)
    log("Рисунок пакета: figures/train_batch.png", report)


def initial_loss(ds: SplitDataset, device: torch.device, report: list[str]) -> None:
    log("", report)
    log("## Початкова втрата L0 (зерно 1, перший пакет, до кроків оптимізатора)", report)
    log("", report)
    log(f"ln 6 = {math.log(6):.4f}", report)
    log("", report)
    log("| Режим | L0 | L0 − ln 6 | std логітів | середня max-ймовірність |", report)
    log("|---|---|---|---|---|", report)
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, 1, device)
        x, y = next(iter(train_loader(ds)))
        set_train_mode(model, regime)
        with torch.no_grad():
            z = model(x.to(device))
            loss = CE(z, y.to(device)).item()
        pmax = torch.softmax(z, 1).max(1).values.mean().item()
        log(f"| {regime} | {loss:.4f} | {loss - math.log(6):+.4f} | {z.std().item():.3f} | {pmax:.3f} |", report)
        del model

def check_params(device: torch.device, report: list[str]) -> None:
    log("", report)
    log("## Параметри режимів", report)
    log("", report)
    log("| Режим | Усього | requires_grad=True | Групи AdamW (параметрів, lr) | Збіг | Заморожені в оптимізаторі |", report)
    log("|---|---|---|---|---|---|", report)
    for regime, cfg in REGIMES.items():
        model, opt = build_model(regime, 1, device)
        total = count(model.parameters())
        trainable = count(p for p in model.parameters() if p.requires_grad)
        groups = [(count(g["params"]), g["lr"]) for g in opt.param_groups]
        names = [n for n, _, _ in cfg["groups"]]
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        frozen_in_opt = sum(1 for p in model.parameters() if not p.requires_grad and id(p) in in_opt)
        all_trainable_in_opt = all(id(p) in in_opt for p in model.parameters() if p.requires_grad)
        ok = (total == EXPECTED_TOTAL and trainable == cfg["expected_trainable"]
              and sum(n for n, _ in groups) == trainable and all_trainable_in_opt)
        gtxt = "; ".join(f"{name}: {n:,} при {lr:g}".replace(",", " ") for name, (n, lr) in zip(names, groups))
        log(f"| {regime} | {total:,} | {trainable:,} | {gtxt} | {'так' if ok else 'НІ'} | {frozen_in_opt} |"
            .replace(",", " "), report)
        if not ok or frozen_in_opt:
            sys.exit(f"Параметри режиму {regime} не збігаються з таблицею режимів.")


def has_bn(module: torch.nn.Module) -> bool:
    return any(isinstance(m, torch.nn.BatchNorm2d) for m in module.modules())



def five_steps(regime: str, ds: SplitDataset, device: torch.device, proper_mode: bool):
    model, opt = build_model(regime, 1, device)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if proper_mode:
        set_train_mode(model, regime)
    else:
        model.train()
    it = iter(train_loader(ds))
    for _ in range(5):
        x, y = next(it)
        opt.zero_grad()
        CE(model(x.to(device)), y.to(device)).backward()
        opt.step()
    return model, before


def module_changes(model: torch.nn.Module, before: dict[str, torch.Tensor]):
    after = model.state_dict()
    result = {}
    for name in TOP_MODULES:
        mod = getattr(model, name)
        params = [f"{name}.{n}" for n, _ in mod.named_parameters()]
        p_changed = any(not torch.equal(before[k], after[k]) for k in params)
        if has_bn(mod):
            bufs = [k for k in after if k.startswith(f"{name}.")
                    and k.endswith(("running_mean", "running_var"))]
            b_changed = any(not torch.equal(before[k], after[k]) for k in bufs)
        else:
            b_changed = None
        result[name] = (p_changed, b_changed)
    return result


def check_changes(ds: SplitDataset, device: torch.device, report: list[str]) -> None:
    log("", report)
    log("## Що змінюється за 5 кроків (set_train_mode, зерно 1)", report)
    log("", report)
    log("| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |", report)
    log("|---|---|---|---|---|", report)
    all_ok = True
    for regime in ("frozen", "unfreeze"):
        model, before = five_steps(regime, ds, device, proper_mode=True)
        frozen = set(REGIMES[regime]["eval_modules"])
        for name, (p, b) in module_changes(model, before).items():
            exp_p = name not in frozen
            exp_b = None if b is None else name not in frozen
            ok = p == exp_p and b == exp_b
            all_ok &= ok
            btxt = "немає BN" if b is None else ("так" if b else "ні")
            log(f"| {regime} | {name} | {'так' if p else 'ні'} | {btxt} | {'так' if ok else 'НІ'} |", report)
    log("", report)
    log(f"Усі модулі поводяться як очікувано: {'так' if all_ok else 'НІ'}", report)
    if not all_ok:
        sys.exit("Перевірка режимів навчання не пройшла.")


PREDICTION = (
    "Прогноз (записано до запуску): з model.train() замість set_train_mode параметри "
    "замороженого кістяка не зміняться - вони мають requires_grad=False і їх немає в "
    "оптимізаторі. Але всі BN кістяка (bn1, layer1-layer4) у режимі train() на кожному "
    "прямому проході оновлять running_mean і running_var ковзним середнім статистик "
    "наших пакетів, тож буфери зміняться у всіх модулях з BN. Кістяк перестане бути "
    "замороженим: у eval() ознаки рахуватимуться за іншими статистиками, ніж в ImageNet. "
    "Із кривої втрати на train цього не видно: у train() BN нормує за статистиками "
    "поточного пакета, а не за ковзними, тому втрата на train виглядатиме нормально; "
    "різниця проявиться лише в eval() на validation/test."
)


def bn_bug_experiment(ds: SplitDataset, device: torch.device, report: list[str]) -> None:
    log("", report)
    log("## Дослід із навмисною помилкою: frozen з model.train()", report)
    log("", report)
    log(PREDICTION, report)
    log("", report)
    model, before = five_steps("frozen", ds, device, proper_mode=False)
    log("| Модуль | Параметри змінилися | Ковзні статистики BN змінилися |", report)
    log("|---|---|---|", report)
    for name, (p, b) in module_changes(model, before).items():
        btxt = "немає BN" if b is None else ("так" if b else "ні")
        log(f"| {name} | {'так' if p else 'ні'} | {btxt} |", report)
    after = model.state_dict()
    k = "layer1.0.bn1"
    dm = (after[f"{k}.running_mean"] - before[f"{k}.running_mean"]).abs().max().item()
    dv = (after[f"{k}.running_var"] - before[f"{k}.running_var"]).abs().max().item()
    nb = int(after[f"{k}.num_batches_tracked"] - before[f"{k}.num_batches_tracked"])
    log("", report)
    log(f"layer1.0.bn1: max |Δ running_mean| = {dm:.6g}, max |Δ running_var| = {dv:.6g}, "
        f"num_batches_tracked +{nb}", report)

def run_one(regime: str, seed: int, train_ds, val_ds, device: torch.device, split_hash: str) -> dict:
    out = RUNS / f"{regime}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model, opt = build_model(regime, seed, device)
    loader = train_loader(train_ds)
    vloader = eval_loader(val_ds)
    y_val = np.array(val_ds.labels)
    history, train_time = [], 0.0
    t_all = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        set_train_mode(model, regime)
        total, steps = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = CE(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        logits = logits_of(model, vloader, device)
        f1 = macro_f1(y_val, logits)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "val_macro_f1": f1,
                        "steps": steps})
        print(f"  {regime} seed {seed} | епоха {epoch}/{EPOCHS} | втрата {total / len(train_ds):.4f} "
              f"| macro-F1 val {f1:.4f} | {time.perf_counter() - t_all:.0f} с", flush=True)
    np.save(out / "val_logits.npy", logits.astype(np.float32))
    torch.save(model.state_dict(), out / "model.pt")
    record = {
        "regime": regime,
        "seed": seed,
        "weights": REGIMES[regime]["weights"],
        "trainable_modules": REGIMES[regime]["trainable"],
        "eval_modules_during_training": REGIMES[regime]["eval_modules"],
        "optimizer_groups": [{"name": n, "modules": m, "lr": lr} for n, m, lr in REGIMES[regime]["groups"]],
        "trainable_params": count(p for p in model.parameters() if p.requires_grad),
        "total_params": count(model.parameters()),
        "settings": settings(),
        "git": git_state(),
        "split_csv_sha256": split_hash,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
        "device": device_name(device),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_macro_f1": history[-1]["val_macro_f1"],
        "train_time_s": train_time,
        "train_plus_val_time_s": time.perf_counter() - t_all,
    }
    (out / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record

def select_and_save(records: dict[tuple[str, int], dict]) -> None:
    by_regime: dict[str, list[dict]] = {}
    for (regime, _), r in records.items():
        by_regime.setdefault(regime, []).append(r)
    means = {g: float(np.mean([r["final_val_macro_f1"] for r in rs])) for g, rs in by_regime.items()}
    regime = max(means, key=means.get)
    best = max(by_regime[regime], key=lambda r: r["final_val_macro_f1"])
    state = torch.load(RUNS / f"{regime}_seed{best['seed']}" / "model.pt", map_location="cpu")
    torch.save({"state_dict": state, "regime": regime, "seed": best["seed"], "classes": CLASSES,
                "img_size": 160}, MODEL_FILE)
    print(f"\nСередні macro-F1 на validation: " + ", ".join(f"{g} {m:.4f}" for g, m in means.items()))
    print(f"Вибрано: {regime}, зерно {best['seed']} (macro-F1 {best['final_val_macro_f1']:.4f}) -> {MODEL_FILE.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека, де лежать seg_train/ і seg_test/")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda або mps")
    ap.add_argument("--skip-existing", action="store_true",
                    help="не перенавчати запуски, для яких уже є run.json і model.pt (продовження після перерви)")
    args = ap.parse_args()
    device = pick_device(args.device)
    for d in (RUNS, FIGURES, CHECKS):
        d.mkdir(exist_ok=True)
    print(f"Пристрій: {device_name(device)}")

    report: list[str] = []
    rows = check_split(args.data, report)
    train_ds = SplitDataset(args.data, rows, "train", train_transform())
    val_ds = SplitDataset(args.data, rows, "validation", eval_transform())
    save_batch_figure(train_ds, report)
    initial_loss(train_ds, device, report)
    check_params(device, report)
    check_changes(train_ds, device, report)
    bn_bug_experiment(train_ds, device, report)
    (CHECKS / "checks.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nПеревірки збережено: checks/checks.md\n")

    split_hash = sha256(SPLIT_CSV)
    records = {}
    for regime, seed in RUN_PLAN:
        d = RUNS / f"{regime}_seed{seed}"
        if args.skip_existing and (d / "run.json").exists() and (d / "model.pt").exists():
            rec = json.loads((d / "run.json").read_text(encoding="utf-8"))
            if rec["split_csv_sha256"] == split_hash:
                print(f"Пропускаю {d.name}: уже навчено")
                records[(regime, seed)] = rec
                continue
        print(f"\nЗапуск {regime}, зерно {seed}")
        records[(regime, seed)] = run_one(regime, seed, train_ds, val_ds, device, split_hash)
    select_and_save(records)


if __name__ == "__main__":
    main()
