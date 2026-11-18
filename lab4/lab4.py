from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
import torch
import torchvision
import torchvision.transforms as T
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from sklearn.metrics import f1_score

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EXPECTED_SPLIT = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
EXPECTED_CALIB = {"n": 200, "sum": 1_921_470,
                  "classes": {"buildings": 40, "forest": 35, "glacier": 36,
                              "mountain": 33, "sea": 29, "street": 27}}
N_CALIB = 200
QUALITY_BATCH = 64
ROUNDS, WARMUP, MEASURES = 3, 20, 100

HERE = Path(__file__).resolve().parent


def read_split(path: Path) -> list[dict]:
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    assert header == ["file", "label", "part"], header
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def name_sum(rows: list[dict]) -> int:
    return sum(int(Path(r["file"]).stem) for r in rows)


def check_split(rows: list[dict]) -> dict:
    out = {}
    for part, (n_exp, s_exp) in EXPECTED_SPLIT.items():
        sub = [r for r in rows if r["part"] == part]
        n, s = len(sub), name_sum(sub)
        assert (n, s) == (n_exp, s_exp), f"{part}: {n}, {s} замість {n_exp}, {s_exp}"
        out[part] = {"n": n, "name_sum": s}
    print(f"[поділ] train {out['train']}, test {out['test']}: збігається з контролем")
    return out


def image_path(data: Path, row: dict) -> Path:
    folder = "seg_train/seg_train" if row["part"] == "train" else "seg_test/seg_test"
    return data / folder / row["label"] / row["file"]


def load_images(data: Path, rows: list[dict], size: int) -> np.ndarray:
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    x = np.empty((len(rows), 3, size, size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(image_path(data, r)) as im:
            x[i] = tf(im.convert("RGB")).numpy()
    return x


def calibration_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    train = sorted((r for r in rows if r["part"] == "train"), key=lambda r: r["file"])
    assert len(train) == 1200
    idx = np.random.default_rng(0).permutation(len(train))[:N_CALIB]
    calib = [train[i] for i in idx]
    counts = {c: sum(r["label"] == c for r in calib) for c in CLASSES}
    info = {"n": len(calib), "sum": name_sum(calib), "classes": counts}
    assert info == EXPECTED_CALIB, f"калібрувальна вибірка: {info}"
    print(f"[калібрування] {info}: збігається з контролем")
    return calib, info


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert list(ckpt["classes"]) == CLASSES, ckpt["classes"]
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: ckpt[k] for k in ("regime", "seed", "classes", "img_size")}
    return model, meta


def export(model: torch.nn.Module, size: int, path: Path) -> float:
    dummy = torch.zeros(1, 3, size, size)
    t0 = time.perf_counter()
    torch.onnx.export(model, (dummy,), str(path), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    return time.perf_counter() - t0


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return np.concatenate([sess.run(["logits"], {"input": x[i:i + QUALITY_BATCH]})[0]
                           for i in range(0, len(x), QUALITY_BATCH)])


@torch.no_grad()
def torch_logits(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    return np.concatenate([model(torch.from_numpy(x[i:i + QUALITY_BATCH])).numpy()
                           for i in range(0, len(x), QUALITY_BATCH)])


def op_counts(path: Path) -> dict:
    nodes = onnx.load(str(path)).graph.node
    return {op: sum(n.op_type == op for n in nodes) for op in ("Conv", "BatchNormalization")}


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro"))


class Reader(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.x = x
        self.i = 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        item = {"input": self.x[self.i:self.i + 1]}
        self.i += 1
        return item


def quantize(prep: Path, out: Path, calib_x: np.ndarray, per_channel: bool) -> float:
    t0 = time.perf_counter()
    quantize_static(str(prep), str(out), Reader(calib_x),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    calibrate_method=CalibrationMethod.MinMax)
    return time.perf_counter() - t0


def quality_row(y: np.ndarray, pred: np.ndarray, pred_fp32: np.ndarray) -> dict:
    n = len(y)
    ok, ok32 = pred == y, pred_fp32 == y
    b = int(np.sum(ok & ~ok32))
    c = int(np.sum(~ok & ok32))
    acc = float(ok.mean())
    return {"macro_f1": macro_f1(y, pred), "accuracy": acc,
            "agreement": float(np.mean(pred == pred_fp32)),
            "b": b, "c": c, "delta_acc": (b - c) / n, "se_delta": math.sqrt(b + c) / n,
            "differs_from_noise": bool(abs(b - c) > 2 * math.sqrt(b + c))}


def channel_ranges(path: Path) -> dict:
    g = onnx.load(str(path)).graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    convs = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        w = inits[node.input[1]]
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        convs.append({"node": node.name, "shape": list(w.shape), "r": r, "bits": math.log2(r)})
    rs = [c["r"] for c in convs]
    med = float(np.median(rs))
    top = sorted(convs, key=lambda c: -c["r"])[:3]
    return {"n_conv": len(convs), "convs": convs, "top3": top,
            "median_r": med, "median_bits": math.log2(med)}


def measure(sess: ort.InferenceSession, x: np.ndarray) -> list[float]:
    feed = {"input": x}
    for _ in range(WARMUP):
        sess.run(["logits"], feed)
    times = []
    for _ in range(MEASURES):
        t0 = time.perf_counter()
        sess.run(["logits"], feed)
        times.append(time.perf_counter() - t0)
    return times


def latency(sessions: dict, x: np.ndarray) -> dict:
    raw = {name: [] for name in sessions}
    for _ in range(ROUNDS):
        for name, sess in sessions.items():
            raw[name].append(measure(sess, x))
    out = {}
    for name, rounds in raw.items():
        ms = np.array(rounds) * 1e3
        out[name] = {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
                     "p50_rounds_ms": [float(np.percentile(r, 50)) for r in ms],
                     "all_ms": ms.tolist()}
    a, b = out["fp32"]["p50_rounds_ms"], out["int8"]["p50_rounds_ms"]
    out["speedup_p50"] = out["fp32"]["p50_ms"] / out["int8"]["p50_ms"]
    out["measured"] = bool(max(a) < min(b) or max(b) < min(a))
    return out


def cpu_model() -> str:
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    return {"cpu": cpu_model(), "machine": platform.machine(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "intra_op_num_threads": 1, "inter_op_num_threads": 1, "batch_size": 1,
            "providers": ["CPUExecutionProvider"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE.parent / "lab3" / "lab3_model.pt")
    ap.add_argument("--split", type=Path, default=HERE.parent / "lab3" / "split.csv")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--ranges-only", action="store_true",
                    help="лише експорт і діапазони каналів (для прогнозу до запуску досліду)")
    args = ap.parse_args()
    out = args.out
    fp32, prep = out / "lab4_fp32.onnx", out / "lab4_fp32_prep.onnx"
    int8, int8_pt = out / "lab4_int8.onnx", out / "lab4_int8_per_tensor.onnx"
    torch.set_num_threads(1)
    res: dict = {"environment": environment()}

    rows = read_split(args.split)
    res["split"] = check_split(rows)
    model, meta = load_model(args.model)
    size = int(meta["img_size"])
    res["model"] = {**meta, "path": str(args.model)}

    t_export = export(model, size, fp32)
    ranges = channel_ranges(fp32)
    if args.ranges_only:
        for c in ranges["top3"]:
            print(f"  r = {c['r']:.2f}  {c['node']}")
        print(f"  медіана r = {ranges['median_r']:.2f} ({ranges['median_bits']:.2f} біта)")
        return

    test = [r for r in rows if r["part"] == "test"]
    y = np.array([CLASSES.index(r["label"]) for r in test])
    x_test = load_images(args.data, test, size)
    z_torch = torch_logits(model, x_test)
    s32 = session(fp32)
    z_onnx = ort_logits(s32, x_test)
    p_torch, p32 = z_torch.argmax(1), z_onnx.argmax(1)
    res["export"] = {"time_s": t_export,
                     "max_abs_logit_diff": float(np.abs(z_onnx - z_torch).max()),
                     "argmax_agreement": float(np.mean(p_torch == p32)),
                     "macro_f1_pytorch": macro_f1(y, p_torch), "macro_f1_onnx_fp32": macro_f1(y, p32),
                     "ops": op_counts(fp32)}
    e = res["export"]
    print(f"[експорт] {t_export:.2f} с; max|Δz| = {e['max_abs_logit_diff']:.2e}; "
          f"збіг класів {e['argmax_agreement']:.4f}; macro-F1 {e['macro_f1_pytorch']:.4f} / "
          f"{e['macro_f1_onnx_fp32']:.4f}; вузли {e['ops']}")
    assert e["max_abs_logit_diff"] < 1e-3 and e["argmax_agreement"] == 1.0
    assert e["ops"] == {"Conv": 20, "BatchNormalization": 0}

    calib, res["calibration"] = calibration_rows(rows)
    res["calibration"]["files"] = [r["file"] for r in calib]
    calib_x = load_images(args.data, calib, size)
    t0 = time.perf_counter()
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    t_prep = time.perf_counter() - t0
    t_q = quantize(prep, int8, calib_x, per_channel=True)
    res["quantization"] = {"pre_process_s": t_prep, "quantize_static_s": t_q}
    print(f"[квантизація] pre_process {t_prep:.2f} с, quantize_static {t_q:.2f} с")

    s8 = session(int8)
    p8 = ort_logits(s8, x_test).argmax(1)
    res["quality"] = {"fp32": quality_row(y, p32, p32), "int8": quality_row(y, p8, p32)}
    acc32 = res["quality"]["fp32"]["accuracy"]
    res["quality"]["se_single_model_fp32"] = math.sqrt(acc32 * (1 - acc32) / len(y))

    t_pt = quantize(prep, int8_pt, calib_x, per_channel=False)
    p_pt = ort_logits(session(int8_pt), x_test).argmax(1)
    res["quality"]["int8_per_tensor"] = quality_row(y, p_pt, p32)
    res["per_tensor"] = {"quantize_static_s": t_pt, **ranges}
    for k, q in res["quality"].items():
        if isinstance(q, dict):
            print(f"[якість] {k:16s} F1 {q['macro_f1']:.4f} acc {q['accuracy']:.4f} "
                  f"згода {q['agreement']:.4f} b={q['b']} c={q['c']} "
                  f"Δacc={q['delta_acc']:+.4f}±{q['se_delta']:.4f} шум? {not q['differs_from_noise']}")

    x1 = np.ascontiguousarray(x_test[:1])
    res["latency"] = latency({"fp32": s32, "int8": s8}, x1)
    res["latency"]["input_file"] = test[0]["file"]
    res["size_mb"] = {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6,
                      "int8_per_tensor": int8_pt.stat().st_size / 1e6}
    L = res["latency"]
    for k in ("fp32", "int8"):
        print(f"[затримка] {k}: {res['size_mb'][k]:.2f} МБ, p50 {L[k]['p50_ms']:.2f} мс, "
              f"p95 {L[k]['p95_ms']:.2f} мс, раунди {[round(v, 2) for v in L[k]['p50_rounds_ms']]}")
    print(f"[затримка] прискорення p50 {L['speedup_p50']:.2f}, виміряна зміна: {L['measured']}")

    (out / "lab4_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"записано {out / 'lab4_results.json'}")


if __name__ == "__main__":
    main()
