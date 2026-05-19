"""
이미지 인식 모델 테스트셋 정확도 평가
======================================
train.py 와 동일한 분할(seed=42, val_ratio=0.2)을 사용하되
val 셋을 test 셋으로 사용해 최종 정확도를 측정한다.

실행:
    python image_recognition/src/evaluate.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

# ── 경로 ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"
MODEL_DIR     = BASE_DIR / "models"
MODEL_PATH    = MODEL_DIR / "best_model.pth"
CLASS_IDX     = MODEL_DIR / "class_to_idx.json"

# ── 하이퍼파라미터 (train.py 와 동일) ─────────────────
BATCH_SIZE = 16
VAL_RATIO  = 0.2
SEED       = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────
class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_samples(processed_root, raw_root):
    classes = sorted(
        d.name
        for group in processed_root.iterdir() if group.is_dir()
        for d in group.iterdir()              if d.is_dir()
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    samples = []
    for group in processed_root.iterdir():
        if not group.is_dir():
            continue
        for product in group.iterdir():
            if not product.is_dir():
                continue
            label  = class_to_idx[product.name]
            n_orig = len([f for f in (raw_root / group.name / product.name).iterdir()
                          if f.suffix.lower() in EXTS])
            for img_path in sorted(product.glob("*.png")):
                file_idx = int(img_path.stem)
                source   = file_idx if file_idx < n_orig else (file_idx - n_orig) % n_orig
                samples.append((img_path, label, product.name, source))

    return samples, classes, class_to_idx


def split_samples(samples, val_ratio, seed):
    groups = defaultdict(list)
    for path, label, product, source in samples:
        groups[(product, source)].append((path, label))

    group_keys = sorted(groups.keys())
    random.Random(seed).shuffle(group_keys)
    n_val    = max(1, int(len(group_keys) * val_ratio))
    val_keys = set(group_keys[:n_val])

    train_s, test_s = [], []
    for key, items in groups.items():
        (test_s if key in val_keys else train_s).extend(items)
    return train_s, test_s


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 데이터 준비
    samples, classes, class_to_idx = build_samples(PROCESSED_DIR, RAW_DIR)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    _, test_samples = split_samples(samples, VAL_RATIO, SEED)

    print(f"테스트셋: {len(test_samples)}장  |  클래스: {len(classes)}개\n")

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_loader = DataLoader(
        ProductDataset(test_samples, test_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # 모델 로드
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 평가
    correct_per_class = defaultdict(int)
    total_per_class   = defaultdict(int)
    total_correct = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, label in zip(preds, labels):
                total_per_class[label.item()]   += 1
                correct_per_class[label.item()] += int(pred == label)
            total_correct += (preds == labels).sum().item()

    # 결과 출력
    print("=" * 55)
    print("클래스별 정확도")
    print("=" * 55)
    for idx in sorted(total_per_class):
        total   = total_per_class[idx]
        correct = correct_per_class[idx]
        print(f"  {idx_to_class[idx]:<45} {correct:>2}/{total}  {correct/total*100:5.1f}%")

    overall = total_correct / len(test_samples)
    print("=" * 55)
    print(f"전체 정확도: {total_correct}/{len(test_samples)}  →  {overall*100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
