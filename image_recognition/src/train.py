"""
EfficientNet-B0 파인튜닝
========================
입력  : image_recognition/data/processed/{GROUP}/{PRODUCT}/
출력  : image_recognition/models/best_model.pth
        image_recognition/models/class_to_idx.json

실행  :
    pip install torch torchvision
    python image_recognition/src/train.py

학습 전략 :
    Phase 1 - 백본 고정, 분류 헤드만 학습 (5 epochs)
    Phase 2 - 전체 파인튜닝, 낮은 lr + CosineAnnealing (15 epochs)
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ── 경로 ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR     = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── 하이퍼파라미터 ────────────────────────────────────
BATCH_SIZE  = 16
EPOCHS_HEAD = 5
EPOCHS_FULL = 15
LR_HEAD     = 1e-3
LR_FULL     = 1e-4
VAL_RATIO   = 0.2
SEED        = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────
class ProductDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples   = samples  # [(Path, int), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_samples(root: Path):
    """processed/ 아래 제품 폴더를 클래스로 매핑해 (path, label) 목록 반환."""
    classes = sorted(
        d.name
        for group in root.iterdir() if group.is_dir()
        for d in group.iterdir()    if d.is_dir()
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for group in root.iterdir():
        if not group.is_dir():
            continue
        for product in group.iterdir():
            if not product.is_dir():
                continue
            label = class_to_idx[product.name]
            for img_path in sorted(product.glob("*.png")):
                samples.append((img_path, label))

    return samples, classes, class_to_idx


def split_samples(samples: list, val_ratio: float, seed: int):
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    return [samples[i] for i in idx[n_val:]], [samples[i] for i in idx[:n_val]]


# ── 학습/검증 한 epoch ────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── 메인 ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # 데이터 준비
    samples, classes, class_to_idx = build_samples(PROCESSED_DIR)
    print(f"클래스 {len(classes)}개  |  전체 {len(samples)}장")
    for i, c in enumerate(classes):
        print(f"  {i:2d}. {c}")

    train_samples, val_samples = split_samples(samples, VAL_RATIO, SEED)
    print(f"\ntrain {len(train_samples)}장  |  val {len(val_samples)}장")

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(
        ProductDataset(train_samples, train_tf),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        ProductDataset(val_samples, val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    # 모델: EfficientNet-B0 + 헤드 교체
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    criterion    = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # ── Phase 1: 백본 고정, 헤드만 학습 ──────────────
    print(f"\n{'='*55}")
    print(f"Phase 1 — 헤드 학습  ({EPOCHS_HEAD} epochs, lr={LR_HEAD})")
    print(f"{'='*55}")

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)

    for epoch in range(1, EPOCHS_HEAD + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, False)

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            saved = " *"

        print(f"  [{epoch:02d}/{EPOCHS_HEAD}]  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}{saved}")

    # ── Phase 2: 전체 파인튜닝 ────────────────────────
    print(f"\n{'='*55}")
    print(f"Phase 2 — 전체 파인튜닝  ({EPOCHS_FULL} epochs, lr={LR_FULL})")
    print(f"{'='*55}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    for epoch in range(1, EPOCHS_FULL + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            saved = " *"

        print(f"  [{epoch:02d}/{EPOCHS_FULL}]  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}{saved}")

    # class_to_idx 저장 (추론 시 필요)
    with open(MODEL_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print("학습 완료")
    print(f"  최고 val acc : {best_val_acc:.3f}")
    print(f"  모델 저장   : {(MODEL_DIR / 'best_model.pth').resolve()}")
    print(f"  클래스 저장 : {(MODEL_DIR / 'class_to_idx.json').resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
