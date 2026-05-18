"""
2단계 인식 파이프라인
====================
Stage 1 : EfficientNet  -> 그룹 분류  (BTS / NEWJEANS / ...)
Stage 2 : FAISS 유사도  -> 제품명 1개 반환

실행:
    python image_recognition/src/predict_faiss.py <이미지 경로>
"""

import sys
import io
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


# ── 모델 로더 ─────────────────────────────────────────
class EfficientNetEmbedder(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.features = base_model.features
        self.avgpool  = base_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


def _load_base(model_path: Path, num_classes: int):
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
    base.load_state_dict(torch.load(model_path, map_location="cpu"))
    return base


def _preprocess(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


# ── Stage 1: 그룹 분류 ────────────────────────────────
def classify_group(classifier: nn.Module, idx_to_class: dict, x: torch.Tensor) -> tuple[str, float]:
    """EfficientNet 으로 가장 확률 높은 클래스 -> 그룹명 추출."""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(x)
        probs  = torch.softmax(logits, dim=1)[0]
    top_idx  = probs.argmax().item()
    top_prob = probs[top_idx].item()
    product  = idx_to_class[top_idx]
    # 제품명은 "GROUP_..." 형식이므로 첫 번째 _ 앞이 그룹
    group = product.split("_")[0]
    return group, top_prob


# ── Stage 2: FAISS 유사도 검색 ───────────────────────
def search_faiss(embedder: EfficientNetEmbedder, group: str, x: torch.Tensor, top_k: int = 3):
    index_path = MODEL_DIR / f"faiss_{group}.index"
    meta_path  = MODEL_DIR / f"faiss_{group}_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS 인덱스 없음: {index_path}\n먼저 build_faiss_db.py 를 실행하세요.")

    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        labels: list[str] = json.load(f)

    embedder.eval()
    with torch.no_grad():
        vec = embedder(x)[0].numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    vec = vec.astype(np.float32).reshape(1, -1)

    distances, indices = index.search(vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((labels[idx], float(dist)))
    return results


# ── 메인 파이프라인 ───────────────────────────────────
def predict(img_path: str, top_k: int = 3):
    with open(MODEL_DIR / "class_to_idx.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    base      = _load_base(MODEL_DIR / "best_model.pth", len(class_to_idx))
    embedder  = EfficientNetEmbedder(base)
    x         = _preprocess(img_path)

    # Stage 1
    group, group_conf = classify_group(base, idx_to_class, x)

    # Stage 2
    results = search_faiss(embedder, group, x, top_k=top_k)

    # 출력
    filename = Path(img_path).name
    print(f"\n[입력]  {filename}")
    print(f"Stage 1  그룹 분류  ->  {group}  ({group_conf:.1%})")
    print(f"Stage 2  FAISS 검색 (상위 {top_k})")
    print("-" * 55)
    for rank, (product, score) in enumerate(results, 1):
        bar = "#" * int(score * 40)
        print(f"  {rank}.  {score:.4f}  {bar:<40}  {product}")

    best_product = results[0][0]
    print(f"\n[최종 결과]  {best_product}\n")
    return best_product


# ── 실행 ─────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    DEFAULT_IMAGE = r"C:\gordeok-AI\test2.png"
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    predict(img_path)
