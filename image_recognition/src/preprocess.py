"""
이미지 전처리 + 증강 스크립트
==============================
목적 : raw 이미지(10장)를 전처리 + 증강하여 학습용 데이터 확보
      제품당 10장 → 50장으로 증강

실행 :
    pip install Pillow
    cd AI/
    python image_recognition/src/preprocess.py

입력 :
    image_recognition/data/raw/
    ├── BND/
    │   ├── BOYNEXTDOOR_2025_SEASONS_GREETINGS/
    │   └── BOYNEXTDOOR_2026_SEASONS_GREETINGS/
    ├── BTS/
    │   ├── BTS_MAP_OF_THE_SOUL_ONE_BLU_RAY/
    │   └── BTS_MEMORIES_OF_2020_BLU_RAY/
    ├── NEWJEANS/
    │   ├── NEWJEANS_2023_SEASONS_GREETINGS/
    │   └── NEWJEANS_2024_SEASONS_GREETINGS/
    ├── SEVENTEEN/
    │   ├── SEVENTEEN_2025_SEASONS_GREETINGS/
    │   └── SEVENTEEN_2026_SEASONS_GREETINGS/
    └── TXT/
        ├── TOMORROWTOGETHER_2026_DECO_KIT/
        └── TOMORROWTOGETHER_2026_SEASONS_GREETINGS/

출력 :
    image_recognition/data/processed/ (동일한 폴더 구조)
    제품당 50장
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance

# ── 경로 설정 ─────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 설정 ─────────────────────────────────────────────
IMG_SIZE     = 224
TARGET_COUNT = 50
EXTS         = {".jpg", ".jpeg", ".png", ".webp"}


# ── 증강 함수 ─────────────────────────────────────────
def augment(img: Image.Image, idx: int) -> Image.Image:
    aug = idx % 5

    if aug == 0:
        # 좌우 반전
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    elif aug == 1:
        # 밝기 조정
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    elif aug == 2:
        # 회전 (-15 ~ +15도)
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    elif aug == 3:
        # 대비 조정
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(factor)

    elif aug == 4:
        # 랜덤 크롭 후 리사이즈
        w, h = img.size
        crop_ratio = random.uniform(0.8, 0.95)
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)
        left  = random.randint(0, w - new_w)
        top   = random.randint(0, h - new_h)
        img   = img.crop((left, top, left + new_w, top + new_h))

    return img


# ── 전처리 함수 ───────────────────────────────────────
def process_image(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


# ── 폴더 처리 ─────────────────────────────────────────
def process_folder(raw_folder: Path, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)

    images = [f for f in raw_folder.iterdir()
              if f.suffix.lower() in EXTS]

    if not images:
        print(f"  ⚠️  이미지 없음: {raw_folder.name}")
        return 0

    print(f"\n  [{raw_folder.name}]")
    print(f"  원본 {len(images)}장 → 목표 {TARGET_COUNT}장")

    count = 0

    # 1. 원본 저장
    for img_path in images:
        try:
            img = Image.open(img_path)
            img = process_image(img)
            img.save(out_folder / f"{count:04d}.png")
            count += 1
        except Exception as e:
            print(f"  ⚠️  원본 처리 실패: {img_path.name} ({e})")

    # 2. 증강으로 목표 수량 채우기
    aug_idx = 0
    while count < TARGET_COUNT:
        src = images[aug_idx % len(images)]
        try:
            img = Image.open(src)
            img = process_image(img)
            img = augment(img, aug_idx)
            img.save(out_folder / f"{count:04d}.png")
            count += 1
        except Exception as e:
            print(f"  ⚠️  증강 실패: {e}")
        aug_idx += 1

    print(f"  → 완료: {count}장 저장")
    return count


# ── 메인 ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("GO르덕 이미지 전처리 + 증강 시작")
    print(f"입력: {RAW_DIR.resolve()}")
    print(f"출력: {PROCESSED_DIR.resolve()}")
    print(f"이미지 크기: {IMG_SIZE}×{IMG_SIZE} | 제품당 목표: {TARGET_COUNT}장")
    print("=" * 55)

    total_products = 0
    total_images   = 0

    for group_dir in sorted(RAW_DIR.iterdir()):
        if not group_dir.is_dir():
            continue

        print(f"\n{'─'*55}")
        print(f"  그룹: {group_dir.name}")
        print(f"{'─'*55}")

        for product_dir in sorted(group_dir.iterdir()):
            if not product_dir.is_dir():
                continue

            out_dir = PROCESSED_DIR / group_dir.name / product_dir.name
            count   = process_folder(product_dir, out_dir)
            total_products += 1
            total_images   += count

    print(f"\n{'='*55}")
    print("전처리 완료")
    print(f"  총 제품 수  : {total_products}개")
    print(f"  총 이미지 수: {total_images}장")
    print(f"  저장 위치   : {PROCESSED_DIR.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()