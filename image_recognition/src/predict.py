"""
학습된 모델로 이미지 예측
실행: python image_recognition/src/predict.py <이미지 경로>
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR  = Path(r"C:\gordeok-AI\image_recognition")
MODEL_DIR = BASE_DIR / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(model_path: Path, num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(img_path: str):
    with open(MODEL_DIR / "class_to_idx.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = load_model(MODEL_DIR / "best_model.pth", len(class_to_idx))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img = Image.open(img_path).convert("RGB")
    x   = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    top5 = probs.topk(5)
    print(f"\n[prediction] {Path(img_path).name}")
    print("-" * 55)
    for prob, idx in zip(top5.values, top5.indices):
        label = idx_to_class[idx.item()]
        bar   = "#" * int(prob.item() * 30)
        print(f"  {prob:.1%}  {bar:<30}  {label}")
    print()


if __name__ == "__main__":
    # VSCode Run 버튼으로 실행 시 여기 경로를 수정하세요
    DEFAULT_IMAGE = r"C:\gordeok-AI\test2.png"

    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    predict(img_path)
