"""Train face emotion model — FER-Autism / FER2013 folder layouts.

Improvements over v1:
  - official train/ test/ splits (no random leak)
  - augment on train
  - ResNet18 transfer (default) or TinyCNN
  - AdamW + cosine LR, early stop, best checkpoint

Usage:
  python code/train_fer.py --data "datasets/FER-Autism/Autism emotion recogition dataset" --epochs 40 --out fer_autism_model.pt
  python code/train_fer.py --data datasets/fer2013_hf --backbone tinycnn --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _discover_splits(root: Path) -> tuple[Path | None, Path | None]:
    train = root / "train" if (root / "train").is_dir() else None
    test = root / "test" if (root / "test").is_dir() else None
    if train and test:
        return train, test
    trains, tests = [], []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        low = p.name.lower()
        if low == "train":
            trains.append(p)
        elif low == "test":
            tests.append(p)
    if trains:
        train = sorted(trains, key=lambda x: len(str(x)))[0]
    if tests:
        test = sorted(tests, key=lambda x: len(str(x)))[0]
    return train, test


def _class_names_from_dir(split_dir: Path) -> list[str]:
    names = []
    for p in sorted(split_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("_"):
            names.append(p.name.lower())
    return names


def _collect_samples(split_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for emo_dir in sorted(split_dir.iterdir()):
        if not emo_dir.is_dir():
            continue
        key = emo_dir.name.lower()
        if key not in class_to_idx:
            continue
        y = class_to_idx[key]
        for img in emo_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTS:
                out.append((img, y))
    return out


def _read_image(path: Path) -> Image.Image:
    data = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if arr is not None:
        return Image.fromarray(arr[:, :, ::-1])
    return Image.open(path).convert("RGB")


class FERDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        *,
        size: int = 96,
        augment: bool = False,
    ):
        self.samples = samples
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _aug(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.35:
            img = img.rotate(random.uniform(-12, 12), resample=Image.BILINEAR)
        if random.random() < 0.35:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.35:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        return img

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = _read_image(path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        if self.augment:
            img = self._aug(img)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        return x, y


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(backbone: str, n_classes: int) -> nn.Module:
    if backbone == "tinycnn":
        return TinyCNN(n_classes)
    if backbone == "resnet18":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(weights=None)
        m.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(m.fc.in_features, n_classes),
        )
        return m
    raise ValueError(f"unknown backbone {backbone}")


def train_one_epoch(model, loader, opt, crit, device, scaler=None):
    model.train()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def class_weights(samples: list[tuple[Path, int]], n_classes: int) -> torch.Tensor:
    counts = Counter(y for _, y in samples)
    total = sum(counts.values())
    w = [total / (n_classes * max(counts.get(i, 1), 1)) for i in range(n_classes)]
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--backbone", choices=["resnet18", "tinycnn"], default="resnet18")
    ap.add_argument("--out", type=str, default="fer_autism_model.pt")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    train_dir, test_dir = _discover_splits(root)
    if not train_dir:
        raise SystemExit(f"no train/ split under {root}")

    classes = _class_names_from_dir(train_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_samples = _collect_samples(train_dir, class_to_idx)
    test_samples = _collect_samples(test_dir, class_to_idx) if test_dir else []

    if not test_samples:
        random.seed(42)
        idx = list(range(len(train_samples)))
        random.shuffle(idx)
        cut = max(1, int(len(idx) * 0.12))
        test_samples = [train_samples[i] for i in idx[:cut]]
        train_samples = [train_samples[i] for i in idx[cut:]]

    train_ds = FERDataset(train_samples, size=args.size, augment=True)
    test_ds = FERDataset(test_samples, size=args.size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, len(classes)).to(device)
    weights = class_weights(train_samples, len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f"classes: {classes}")
    print(f"train={len(train_samples)} test={len(test_samples)} backbone={args.backbone} size={args.size} device={device}")

    best_acc = 0.0
    best_state = None
    stale = 0

    for ep in range(1, args.epochs + 1):
        loss, tacc = train_one_epoch(model, train_loader, opt, crit, device, scaler)
        tloss, acc = eval_loader(model, test_loader, device)
        sched.step()
        print(
            f"epoch {ep}/{args.epochs} train_loss={loss:.4f} train_acc={tacc:.3f} "
            f"test_loss={tloss:.4f} test_acc={acc:.3f} lr={opt.param_groups[0]['lr']:.2e}"
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {ep} (patience {args.patience})")
                break

    if best_state:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path

    payload = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "data_root": str(root),
        "test_acc": best_acc,
        "backbone": args.backbone,
        "img_size": args.size,
        "normalize": "imagenet",
    }
    torch.save(payload, out_path)
    meta = {
        "classes": classes,
        "test_acc": round(best_acc, 4),
        "backbone": args.backbone,
        "img_size": args.size,
        "weights": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out_path} best_test_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
