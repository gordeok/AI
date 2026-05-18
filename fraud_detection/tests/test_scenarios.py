"""
Issue #6 — 사기 탐지 회귀 테스트 (9개 케이스)
서버가 실행 중인 상태에서 pytest로 실행한다.
  $ pytest tests/test_scenarios.py -v
"""

import pytest
import httpx

BASE_URL = "http://localhost:8003/api/fraud/analyze"


def _req(room_id: str, role: str, content: str) -> dict:
    return {
        "room_id": room_id,
        "messages": [
            {"sender_id": "u1", "role": role, "content": content, "timestamp": "2024-01-01T10:00:00"}
        ],
    }


def analyze(room_id: str, role: str, content: str) -> dict:
    res = httpx.post(BASE_URL, json=_req(room_id, role, content), timeout=60)
    res.raise_for_status()
    return res.json()


# ── SAFE ──────────────────────────────────────────────────────────────────────

def test_safe_normal_trade():
    """정상 거래 문의는 SAFE"""
    result = analyze("safe-01", "구매자", "포카 상태 괜찮나요? 직거래 가능한가요?")
    assert result["action"] == "none", f"expected none, got {result}"


# ── DANGER ────────────────────────────────────────────────────────────────────

def test_danger_prepayment():
    """선입금 유도 → DANGER"""
    result = analyze("danger-01", "판매자", "입금 먼저 해주시면 바로 보내드릴게요.")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


def test_danger_external_channel():
    """외부 채널 이동 유도 → DANGER"""
    result = analyze("danger-02", "판매자", "카카오톡 오픈채팅으로 와주세요. 링크 드릴게요.")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


def test_danger_fake_safe_payment():
    """안전결제 사칭 → DANGER"""
    result = analyze("danger-03", "판매자", "안전결제 링크로 결제해주세요. 에스크로 방식이에요.")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


def test_danger_external_url():
    """외부 결제 URL 포함 → DANGER"""
    result = analyze("danger-04", "판매자", "이 링크 들어가서 결제해주세요. http://fake-pay.xyz/abc")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


def test_danger_personal_info():
    """개인정보 요청 → DANGER"""
    result = analyze("danger-05", "판매자", "배송 위해 계좌번호랑 이름 알려주세요.")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


def test_danger_gift_card():
    """문화상품권 유도 → DANGER"""
    result = analyze("danger-06", "판매자", "문화상품권으로 결제해주시면 돼요. 핀번호 보내주세요.")
    assert result["action"] == "show_danger_banner", f"expected danger, got {result}"


# ── WARNING ───────────────────────────────────────────────────────────────────

def test_warning_abnormal_price():
    """시세 대비 급매 → WARNING"""
    result = analyze("warning-01", "판매자", "원가 10만원짜리인데 2만원에 급처합니다. 빨리 연락주세요.")
    assert result["action"] == "show_warning_banner", f"expected warning, got {result}"


def test_warning_vague_condition():
    """모호한 거래 조건 → WARNING"""
    result = analyze("warning-02", "판매자", "조건은 나중에 얘기해요. 일단 자리 맡으시겠어요?")
    assert result["action"] == "show_warning_banner", f"expected warning, got {result}"
