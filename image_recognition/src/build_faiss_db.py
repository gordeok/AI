"""
FAISS DB 구축
=============
학습된 EfficientNet 백본으로 이미지 임베딩 추출 후 그룹별 FAISS 인덱스 저장.

출력 (image_recognition/models/):
    faiss_{GROUP}.index      -- FAISS FlatIP 인덱스 (코사인 유사도)
    faiss_{GROUP}_meta.json  -- 각 벡터에 대응하는 제품명 리스트

실행:
    pip install faiss-cpu
    python image_recognition/src/build_faiss_db.py
"""

import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import faiss
from torchvision import models, transforms
from PIL import Image

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR     = BASE_DIR / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class EfficientNetEmbedder(nn.Module):
    """EfficientNet 분류 헤드를 제거한 1280-dim 임베딩 추출기."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.features = base_model.features
        self.avgpool  = base_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 1280)


def load_embedder(model_path: Path, num_classes: int) -> EfficientNetEmbedder:
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
    base.load_state_dict(torch.load(model_path, map_location="cpu"))
    embedder = EfficientNetEmbedder(base)
    embedder.eval()
    return embedder


def embed_image(embedder: EfficientNetEmbedder, img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x   = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        vec = embedder(x)[0].numpy()
    vec /= (np.linalg.norm(vec) + 1e-8)  # L2 정규화 → 내적 = 코사인 유사도
    return vec.astype(np.float32)


def build():
    with open(MODEL_DIR / "class_to_idx.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    embedder = load_embedder(MODEL_DIR / "best_model.pth", len(class_to_idx))
    print(f"모델 로드 완료  |  클래스 {len(class_to_idx)}개\n")

    # 그룹별로 이미지 수집
    group_data: dict[str, dict] = {}  # group -> {vectors: [], labels: []}
    for group_dir in sorted(PROCESSED_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        group_data[group] = {"vectors": [], "labels": []}

        for product_dir in sorted(group_dir.iterdir()):
            if not product_dir.is_dir():
                continue
            product = product_dir.name
            imgs    = sorted(product_dir.glob("*.png"))
            print(f"  {group}/{product}: {len(imgs)}장 임베딩 중...")

            for img_path in imgs:
                vec = embed_image(embedder, img_path)
                group_data[group]["vectors"].append(vec)
                group_data[group]["labels"].append(product)

    print()

    # 그룹별 FAISS 인덱스 저장
    for group, data in group_data.items():
        vectors = np.stack(data["vectors"])        # (N, 1280)
        dim     = vectors.shape[1]

        index = faiss.IndexFlatIP(dim)             # 내적 = 코사인 유사도 (정규화 전제)
        index.add(vectors)

        faiss.write_index(index, str(MODEL_DIR / f"faiss_{group}.index"))
        with open(MODEL_DIR / f"faiss_{group}_meta.json", "w", encoding="utf-8") as f:
            json.dump(data["labels"], f, ensure_ascii=False)

        print(f"[{group}]  벡터 {len(data['vectors'])}개  ->  faiss_{group}.index 저장")

    print("\nFAISS DB 구축 완료")


if __name__ == "__main__":
    build()
