"""
신규 제품 추가 (재학습 없음)
==============================
이미지 몇 장 + 제품명만으로 FAISS DB에 추가.
새 그룹이면 FAISS 인덱스도 자동 생성.

사용법:
    python image_recognition/src/add_product.py \\
        --product "IVE_2024_SEASONS_GREETINGS" \\
        --group   "IVE" \\
        --images  "C:/img1.jpg" "C:/img2.jpg" "C:/img3.jpg"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import faiss
from torchvision import models, transforms
from PIL import Image

BASE_DIR  = Path(r"C:\gordeok-AI\image_recognition")
MODEL_DIR = BASE_DIR / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EMBED_DIM = 1280  # EfficientNet-B0 출력 차원


class EfficientNetEmbedder(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.features = base_model.features
        self.avgpool  = base_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


def load_embedder() -> EfficientNetEmbedder:
    with open(MODEL_DIR / "class_to_idx.json", encoding="utf-8") as f:
        num_classes = len(json.load(f))
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
    base.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    embedder = EfficientNetEmbedder(base)
    embedder.eval()
    return embedder


def embed_images(embedder: EfficientNetEmbedder, img_paths: list[str]) -> np.ndarray:
    vecs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x   = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            v = embedder(x)[0].numpy()
        v /= (np.linalg.norm(v) + 1e-8)
        vecs.append(v.astype(np.float32))
    return np.stack(vecs)


def add_product(group: str, product: str, img_paths: list[str]):
    embedder  = load_embedder()
    vectors   = embed_images(embedder, img_paths)

    index_path = MODEL_DIR / f"faiss_{group}.index"
    meta_path  = MODEL_DIR / f"faiss_{group}_meta.json"

    # 기존 인덱스가 있으면 로드, 없으면 새로 생성
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            labels: list[str] = json.load(f)
        print(f"기존 [{group}] 인덱스 로드  (벡터 {index.ntotal}개)")
    else:
        index  = faiss.IndexFlatIP(EMBED_DIM)
        labels = []
        print(f"새 [{group}] 인덱스 생성")

    index.add(vectors)
    labels.extend([product] * len(vectors))

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    print(f"[{group}] {product}  +{len(vectors)}장  ->  총 {index.ntotal}개 벡터")
    print("추가 완료. 재학습 없이 바로 인식 가능합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, help="제품명 (예: IVE_2024_SEASONS_GREETINGS)")
    parser.add_argument("--group",   required=True, help="그룹명 (예: IVE)")
    parser.add_argument("--images",  nargs="+", required=True, help="이미지 경로들")
    args = parser.parse_args()

    add_product(args.group, args.product, args.images)
