"""
FastAPI 서버
============
POST /predict  이미지 업로드 -> 그룹 + 제품명 반환

실행:
    pip install fastapi uvicorn python-multipart
    uvicorn image_recognition.api.api:app --reload
    또는
    python image_recognition/api/api.py
"""

import sys
import json
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# src 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict_faiss import (
    _load_base,
    EfficientNetEmbedder,
    _preprocess,
    classify_group,
    search_faiss,
    MODEL_DIR,
)

# ── 모델을 서버 시작 시 한 번만 로드 ─────────────────────
models_cache: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_DIR / "class_to_idx.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    base     = _load_base(MODEL_DIR / "best_model.pth", len(class_to_idx))
    embedder = EfficientNetEmbedder(base)

    models_cache["classifier"]  = base
    models_cache["embedder"]    = embedder
    models_cache["idx_to_class"] = {v: k for k, v in class_to_idx.items()}

    print("모델 로드 완료")
    yield
    models_cache.clear()


app = FastAPI(title="GO르덕 AI", lifespan=lifespan)


# ── 엔드포인트 ────────────────────────────────────────
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 이미지 형식 검사
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="jpg, png, webp 파일만 가능합니다.")

    # 임시 파일로 저장 후 예측
    suffix = Path(image.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        x = _preprocess(tmp_path)

        # Stage 1: 그룹 분류
        group, group_conf = classify_group(
            models_cache["classifier"],
            models_cache["idx_to_class"],
            x,
        )

        # Stage 2: FAISS 유사도 검색
        results = search_faiss(models_cache["embedder"], group, x, top_k=1)
        product, similarity = results[0]

    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "group":            group,
        "product":          product,
        "group_confidence": round(group_conf, 4),
        "similarity_score": round(similarity, 4),
        "is_confident":     similarity >= 0.6,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 직접 실행 ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
