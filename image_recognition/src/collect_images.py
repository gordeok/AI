"""
아이돌 시즌그리팅·블루레이·DVD 이미지 수집 스크립트
=====================================================
목적 : GO르덕 이미지 인식 AI MVP용 학습 데이터 수집
      제품명 레이블 포함하여 FAISS DB 구축에 활용

실행 :
    pip install requests python-dotenv
    cd AI/
    python image_recognition/src/collect_images.py

출력 :
    image_recognition/data/raw/
    ├── BTS/
    │   ├── BTS_MEMORIES_OF_2020_BLURAY/               (목표 25장)
    │   └── BTS_MAP_OF_THE_SOUL_ONE_BLURAY/            (목표 25장)
    ├── TXT/
    │   ├── TXT_2026_SEASONS_GREETINGS/                (목표 25장)
    │   └── TXT_2026_DECO_KIT/                         (목표 25장)
    ├── BOYNEXTDOOR/
    │   ├── BND_2025_SEASONS_GREETINGS/                (목표 25장)
    │   └── BND_2026_SEASONS_GREETINGS/                (목표 25장)
    ├── ENHYPEN/
    │   ├── ENHYPEN_2026_SEASONS_GREETINGS_SET/        (목표 25장)
    │   └── ENHYPEN_WALK_THE_LINE_BLURAY/              (목표 25장)
    └── SEVENTEEN/
        ├── SEVENTEEN_2025_SEASONS_GREETINGS/          (목표 25장)
        └── SEVENTEEN_2026_SEASONS_GREETINGS/          (목표 25장)

    image_recognition/data/product_labels.json <- 제품명 레이블 매핑
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── 환경변수 로드 ─────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.environ["GOOGLE_API_KEY"]
CX      = os.environ["GOOGLE_CX"]

# ── 경로 설정 ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # image_recognition/
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 제품 목록 ─────────────────────────────────────────
PRODUCTS = [

    # ── BTS (방탄소년단) ───────────────────────────────
    {
        "group"      : "BTS",
        "product"    : "BTS MEMORIES OF 2020 BLU-RAY",
        "folder_name": "BTS_MEMORIES_OF_2020_BLURAY",
        "target"     : 25,
        "queries"    : [
            "BTS MEMORIES OF 2020 BLU-RAY",
            "방탄소년단 MEMORIES OF 2020 블루레이",
        ],
    },
    {
        "group"      : "BTS",
        "product"    : "BTS MAP OF THE SOUL ON:E BLU-RAY",
        "folder_name": "BTS_MAP_OF_THE_SOUL_ONE_BLURAY",
        "target"     : 25,
        "queries"    : [
            "BTS MAP OF THE SOUL ON:E BLU-RAY",
            "방탄소년단 MAP OF THE SOUL ON:E 블루레이",
        ],
    },

    # ── TXT (투모로우바이투게더) ───────────────────────
    {
        "group"      : "TXT",
        "product"    : "TOMORROW X TOGETHER 2026 SEASON'S GREETINGS",
        "folder_name": "TXT_2026_SEASONS_GREETINGS",
        "target"     : 25,
        "queries"    : [
            "TOMORROW X TOGETHER 2026 SEASON'S GREETINGS",
            "투모로우바이투게더 2026 시즌그리팅",
        ],
    },
    {
        "group"      : "TXT",
        "product"    : "TOMORROW X TOGETHER 2026 DECO KIT",
        "folder_name": "TXT_2026_DECO_KIT",
        "target"     : 25,
        "queries"    : [
            "TOMORROW X TOGETHER 2026 DECO KIT",
            "투모로우바이투게더 2026 데코킷",
        ],
    },

    # ── BOYNEXTDOOR (보이넥스트도어) ──────────────────
    {
        "group"      : "BOYNEXTDOOR",
        "product"    : "BOYNEXTDOOR 2025 SEASON'S GREETINGS",
        "folder_name": "BND_2025_SEASONS_GREETINGS",
        "target"     : 25,
        "queries"    : [
            "BOYNEXTDOOR 2025 SEASON'S GREETINGS",
            "보이넥스트도어 2025 시즌그리팅",
        ],
    },
    {
        "group"      : "BOYNEXTDOOR",
        "product"    : "BOYNEXTDOOR 2026 SEASON'S GREETINGS",
        "folder_name": "BND_2026_SEASONS_GREETINGS",
        "target"     : 25,
        "queries"    : [
            "BOYNEXTDOOR 2026 SEASON'S GREETINGS",
            "보이넥스트도어 2026 시즌그리팅",
        ],
    },

    # ── ENHYPEN (엔하이픈) ────────────────────────────
    {
        "group"      : "ENHYPEN",
        "product"    : "ENHYPEN 2026 SEASON'S GREETINGS / 2026 GGU GGU BOOK (SET)",
        "folder_name": "ENHYPEN_2026_SEASONS_GREETINGS_SET",
        "target"     : 25,
        "queries"    : [
            "ENHYPEN 2026 SEASON'S GREETINGS GGU GGU BOOK",
            "엔하이픈 2026 시즌그리팅 꾸꾸북",
        ],
    },
    {
        "group"      : "ENHYPEN",
        "product"    : "ENHYPEN WORLD TOUR 'WALK THE LINE' IN GOYANG",
        "folder_name": "ENHYPEN_WALK_THE_LINE_BLURAY",
        "target"     : 25,
        "queries"    : [
            "ENHYPEN WORLD TOUR WALK THE LINE IN GOYANG",
            "엔하이픈 WALK THE LINE 고양 블루레이",
        ],
    },

    # ── SEVENTEEN (세븐틴) ────────────────────────────
    {
        "group"      : "SEVENTEEN",
        "product"    : "SEVENTEEN 2025 SEASON'S GREETINGS + WALL CALENDAR SET",
        "folder_name": "SEVENTEEN_2025_SEASONS_GREETINGS",
        "target"     : 25,
        "queries"    : [
            "SEVENTEEN 2025 SEASON'S GREETINGS WALL CALENDAR",
            "세븐틴 2025 시즌그리팅 월캘린더",
        ],
    },
    {
        "group"      : "SEVENTEEN",
        "product"    : "SEVENTEEN 2026 SEASON'S GREETINGS + WALL CALENDAR SET",
        "folder_name": "SEVENTEEN_2026_SEASONS_GREETINGS",
        "target"     : 25,
        "queries"    : [
            "SEVENTEEN 2026 SEASON'S GREETINGS WALL CALENDAR",
            "세븐틴 2026 시즌그리팅 월캘린더",
        ],
    },
]


# ── Google Custom Search API 이미지 URL 검색 ──────────
def search_images(query: str, num: int) -> list:
    urls = []
    for start in range(1, num + 1, 10):
        batch = min(10, num - len(urls))
        if batch <= 0:
            break
        try:
            res = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key"       : API_KEY,
                    "cx"        : CX,
                    "q"         : query,
                    "searchType": "image",
                    "num"       : batch,
                    "start"     : start,
                },
                timeout=10,
            )
            res.raise_for_status()
            for item in res.json().get("items", []):
                urls.append(item["link"])
        except Exception as e:
            print(f"    ⚠️  검색 오류: {e}")
            break
        time.sleep(0.5)
    return urls


# ── 이미지 다운로드 ───────────────────────────────────
def download_image(url: str, path: Path) -> bool:
    try:
        res = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        res.raise_for_status()
        path.write_bytes(res.content)
        return True
    except Exception:
        return False


def count_images(directory: Path) -> int:
    return sum(len(list(directory.glob(f"*.{ext}")))
               for ext in ["jpg", "jpeg", "png", "webp"])


# ── 제품별 수집 ───────────────────────────────────────
def collect_product(product: dict) -> int:
    group       = product["group"]
    folder_name = product["folder_name"]
    prod_name   = product["product"]
    target      = product["target"]
    queries     = product["queries"]

    save_dir = RAW_DIR / group / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    per_query = -(-target // len(queries))  # ceiling division

    print(f"\n  [{prod_name}]")
    print(f"  목표 {target}장 | 쿼리 {len(queries)}개 | 쿼리당 최대 {per_query}장")

    for i, query in enumerate(queries, 1):
        current = count_images(save_dir)
        if current >= target:
            print(f"  목표 달성 ({current}장) — 나머지 쿼리 스킵")
            break

        num = min(per_query, target - current)
        print(f"  [{i}/{len(queries)}] \"{query}\" -> {num}장 시도")

        urls = search_images(query, num)
        downloaded = 0
        for url in urls:
            ext = url.split(".")[-1].split("?")[0][:4].lower()
            if ext not in ("jpg", "jpeg", "png", "webp"):
                ext = "jpg"
            idx = count_images(save_dir)
            if download_image(url, save_dir / f"{idx:04d}.{ext}"):
                downloaded += 1

        print(f"    -> {downloaded}장 다운로드")
        time.sleep(1)

    final = count_images(save_dir)
    print(f"  -> 최종: {final}장 / 목표 {target}장")
    return final


# ── 레이블 JSON 저장 ──────────────────────────────────
def save_labels():
    labels = {
        p["folder_name"]: {"group": p["group"], "product": p["product"]}
        for p in PRODUCTS
    }
    label_path = BASE_DIR / "data" / "product_labels.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"  레이블 저장 -> {label_path}")


# ── 메인 ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("GO르덕 이미지 수집 시작")
    print(f"저장 경로: {RAW_DIR.resolve()}")
    print(f"총 제품 수: {len(PRODUCTS)}개")
    print("=" * 55)

    save_labels()

    summary = {}
    current_group = None

    for product in PRODUCTS:
        if product["group"] != current_group:
            current_group = product["group"]
            print(f"\n{'─'*55}")
            print(f"  그룹: {current_group}")
            print(f"{'─'*55}")

        count = collect_product(product)
        summary[product["folder_name"]] = {
            "product"  : product["product"],
            "collected": count,
            "target"   : product["target"],
        }
        time.sleep(2)

    print(f"\n{'='*55}")
    print("수집 완료 요약")
    print(f"{'='*55}")
    total = 0
    for result in summary.values():
        status = "OK" if result["collected"] >= result["target"] else "NG"
        print(f"  [{status}] {result['product']:<50} "
              f"{result['collected']:>3}장 / {result['target']}장")
        total += result["collected"]
    print(f"\n  총 수집: {total}장")
    print(f"  저장 위치: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
