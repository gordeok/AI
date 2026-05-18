"""
2단계: EXAONE3.5:2.4b 대화 맥락 분석
Ollama REST API 호출 + 3단계 JSON 파싱 폴백
"""

import re
import json
import httpx
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5:2.4b"
TIMEOUT    = 30

SYSTEM_PROMPT = """너는 분철 거래 사기 탐지 AI야.
아래 대화에서 사기 징후를 분석하고 JSON만 반환해. 다른 말은 하지 마.

분류 기준:
- DANGER: 선입금 유도, 외부결제 유도, 안전결제 사칭, 외부채널 이동 유도, 개인정보 요청
- WARNING: 모호한 거래 조건, 비정상 가격, 불분명한 구성품 안내
- SAFE: 정상적인 분철 거래 대화

반환 형식 (JSON만, 설명 없이):
{"risk_level": "SAFE|WARNING|DANGER", "reason": "이유", "triggered_message": "해당 메시지 또는 null"}"""


@dataclass
class LLMResult:
    risk_level: str
    reason: str
    triggered_message: str | None


def _build_conversation(messages: list[dict]) -> str:
    return "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in messages)


def _parse(text: str) -> LLMResult:
    # 1단계: 정상 파싱
    try:
        data = json.loads(text.strip())
        return LLMResult(
            risk_level=data.get("risk_level", "SAFE").upper(),
            reason=data.get("reason", ""),
            triggered_message=data.get("triggered_message"),
        )
    except json.JSONDecodeError:
        pass

    # 2단계: {...} 블록 추출
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return LLMResult(
                risk_level=data.get("risk_level", "SAFE").upper(),
                reason=data.get("reason", ""),
                triggered_message=data.get("triggered_message"),
            )
        except json.JSONDecodeError:
            pass

    # 3단계: 키워드 폴백
    upper = text.upper()
    level = "DANGER" if "DANGER" in upper else ("WARNING" if "WARNING" in upper else "SAFE")
    return LLMResult(risk_level=level, reason="파싱 실패 — 키워드 폴백", triggered_message=None)


def analyze(messages: list[dict]) -> LLMResult:
    prompt = f"{SYSTEM_PROMPT}\n\n[대화]\n{_build_conversation(messages)}"
    try:
        res = httpx.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        res.raise_for_status()
        return _parse(res.json().get("response", ""))
    except httpx.TimeoutException:
        return LLMResult(risk_level="SAFE", reason="LLM 타임아웃", triggered_message=None)
    except Exception as e:
        return LLMResult(risk_level="SAFE", reason=f"LLM 오류: {e}", triggered_message=None)
