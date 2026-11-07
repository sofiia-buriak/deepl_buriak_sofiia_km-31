from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parent
SPLIT_CSV = ROOT / "split.csv"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
CHECKS = ROOT / "checks"
MODEL_FILE = ROOT / "lab3_model.pt"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PARTS = ["train", "validation", "test"]
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_TRAIN_PER_CLASS = 200

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

REGIMES = {
    "scratch": {
        "weights": None,
        "trainable": TOP_MODULES,
        "eval_modules": [],
        "groups": [("all", TOP_MODULES, LR_HEAD)],
        "expected_trainable": 11_179_590,
    },
    "frozen": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "groups": [("fc", ["fc"], LR_HEAD)],
        "expected_trainable": 3_078,
    },
    "unfreeze": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["layer4", "fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "groups": [("fc", ["fc"], LR_HEAD), ("layer4", ["layer4"], LR_BACKBONE)],
        "expected_trainable": 8_396_806,
    },
}
EXPECTED_TOTAL = 11_179_590
RUN_PLAN = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
            ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1_996_761), "validation": (218, 4_837_323), "test": (219, 4_837_559)},
    "forest": {"train": (200, 2_032_625), "validation": (237, 5_276_962), "test": (237, 5_291_326)},
    "glacier": {"train": (200, 2_183_923), "validation": (276, 6_135_652), "test": (277, 6_157_250)},
    "mountain": {"train": (200, 1_950_345), "validation": (262, 5_789_535), "test": (263, 5_849_765)},
    "sea": {"train": (200, 1_987_613), "validation": (255, 5_655_068), "test": (255, 5_668_011)},
    "street": {"train": (200, 1_841_817), "validation": (250, 5_541_845), "test": (251, 5_547_533)},
}

def class_dir(data: Path, part: str, label: str) -> Path:
    sub = "seg_train" if part == "train" else "seg_test"
    return data / sub / sub / label


def list_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")


def make_split(data: Path) -> list[dict[str, str]]:
    rows = []
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = list_files(class_dir(data, "train", label))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        rows += [{"file": files[i], "label": label, "part": "train"} for i in sorted(idx[:N_TRAIN_PER_CLASS])]
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = list_files(class_dir(data, "test", label))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        rows += [{"file": files[i], "label": label, "part": "validation"} for i in sorted(idx[:cut])]
        rows += [{"file": files[i], "label": label, "part": "test"} for i in sorted(idx[cut:])]
    order = {p: i for i, p in enumerate(PARTS)}
    rows.sort(key=lambda r: order[r["part"]])
    return rows


def write_split(rows: list[dict[str, str]], path: Path = SPLIT_CSV) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path = SPLIT_CSV) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_control(rows: list[dict[str, str]]) -> dict[str, dict[str, tuple[int, int]]]:
    table = {c: {p: (0, 0) for p in PARTS} for c in CLASSES}
    for r in rows:
        n, s = table[r["label"]][r["part"]]
        table[r["label"]][r["part"]] = (n + 1, s + int(Path(r["file"]).stem))
    return table


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def train_transform() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class SplitDataset(Dataset):

    def __init__(self, data: Path, rows: list[dict[str, str]], part: str, transform):
        self.rows = [r for r in rows if r["part"] == part]
        self.paths = [class_dir(data, part, r["label"]) / r["file"] for r in self.rows]
        self.labels = [CLASSES.index(r["label"]) for r in self.rows]
        self.transform = transform
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def image(self, i: int) -> Image.Image:
        if i not in self._cache:
            with Image.open(self.paths[i]) as im:
                self._cache[i] = im.convert("RGB")
        return self._cache[i]

    def __getitem__(self, i: int):
        return self.transform(self.image(i)), self.labels[i]


def train_loader(ds: SplitDataset) -> DataLoader:
    return DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(ds: SplitDataset) -> DataLoader:
    return DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)

def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    import platform
    return f"CPU ({platform.processor() or platform.machine()})"


def build_model(regime: str, seed: int, device: torch.device):
    cfg = REGIMES[regime]
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if cfg["weights"] is None else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in cfg["trainable"]
    model.to(device=device, dtype=torch.float32)
    groups = []
    for _, modules, lr in cfg["groups"]:
        params = [p for m in modules for p in getattr(model, m).parameters() if p.requires_grad]
        groups.append({"params": params, "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return model, opt


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    model.train()
    for name in REGIMES[regime]["eval_modules"]:
        getattr(model, name).eval()


def count(params) -> int:
    return sum(p.numel() for p in params)


@torch.no_grad()
def logits_of(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y: np.ndarray, logits: np.ndarray) -> float:
    return float(f1_score(y, logits.argmax(1), average="macro", labels=list(range(len(CLASSES)))))

def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "HEAD") or None
    dirty = bool(run("status", "--porcelain", "--untracked-files=no", "--", ".")) if commit else None
    return {"commit": commit, "uncommitted_changes": dirty}


def settings() -> dict[str, object]:
    return {
        "classes": CLASSES,
        "transforms": {
            "train": [
                {"RandomResizedCrop": {"size": IMG_SIZE, "scale": [0.7, 1.0], "ratio": [3 / 4, 4 / 3],
                                       "interpolation": "bilinear", "antialias": True}},
                {"RandomHorizontalFlip": {"p": 0.5}},
                {"ToTensor": {}},
                {"Normalize": {"mean": MEAN, "std": STD}},
            ],
            "validation_test": [
                {"Resize": {"size": [IMG_SIZE, IMG_SIZE], "interpolation": "bilinear", "antialias": True}},
                {"ToTensor": {}},
                {"Normalize": {"mean": MEAN, "std": STD}},
            ],
            "image_open": "PIL.Image.open(...).convert('RGB')",
        },
        "model": {
            "architecture": "torchvision.models.resnet18",
            "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
            "batchnorm": {"momentum": 0.1, "eps": 1e-5},
            "dtype": "float32",
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WEIGHT_DECAY, "amsgrad": False,
            "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
        },
        "loss": "torch.nn.CrossEntropyLoss(), середнє за пакетом",
        "loop": {
            "epochs": EPOCHS,
            "steps_per_epoch": 38,
            "total_steps": 304,
            "train_loader": {"batch_size": TRAIN_BATCH, "shuffle": True, "drop_last": False,
                             "num_workers": 0, "generator": "глобальний генератор PyTorch"},
            "eval_loader": {"batch_size": EVAL_BATCH, "shuffle": False, "num_workers": 0},
            "lr_schedule": None, "grad_clip": None, "amp": False, "label_smoothing": 0.0,
            "class_weights": None, "early_stopping": False, "checkpoint": "ваги після 8-ї епохи",
        },
    }
