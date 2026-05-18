"""
2단계: EXAONE3.5:2.4b 대화 맥락 분석
Ollama REST API 호출 + 3단계 JSON 파싱 폴백
"""

import re
import json
import httpx
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME   = "exaone3.5:2.4b"
TIMEOUT      = 30
WINDOW_SIZE  = 20  # LLM에 넘길 최근 메시지 수
TEMPERATURE  = 0.1  # 낮을수록 일관된 JSON 출력
TOP_P        = 0.9

SYSTEM_PROMPT = """너는 K-pop 분철 거래 사기 탐지 AI야.
아래 대화를 분석하고 JSON만 반환해. 다른 말은 절대 하지 마.

[분류 기준]
DANGER : 선입금/계좌이체 유도, 외부결제(상품권·페이) 유도, 가짜 안전결제 링크, 외부채널(카톡·오픈채팅) 이동 유도, 개인정보 요청
WARNING: 비정상 저가, 구성품 불명확, 거래 조건 반복 변경, 과도한 마감 압박
SAFE   : 플랫폼 결제·배송·구성품 정상 문의

[예시]
대화:
[판매자] 입금 먼저 해주시면 자리 확정해드릴게요.
출력:
{"risk_level":"DANGER","reason":"선입금 유도 — 플랫폼 결제 우회 후 잠수 위험","triggered_message":"입금 먼저 해주시면 자리 확정해드릴게요"}

대화:
[판매자] 문화상품권 핀번호 보내주시면 돼요. 구글 플레이 기프트카드도 괜찮아요.
출력:
{"risk_level":"DANGER","reason":"외부 결제 수단(상품권) 유도 — 환불 불가 구조","triggered_message":"문화상품권 핀번호 보내주시면 돼요"}

대화:
[판매자] 자세한 건 카카오 오픈채팅으로 오세요. 링크 드릴게요.
출력:
{"risk_level":"DANGER","reason":"외부 채널 이동 유도 — 플랫폼 기록 회피","triggered_message":"카카오 오픈채팅으로 오세요"}

대화:
[구매자] 배송은 언제 돼요?
[판매자] 뭐가 오는지는 받아봐야 알죠 하하. 가격은 원래 25,000원인데 배송비 추가돼서 30,000원이에요.
출력:
{"risk_level":"WARNING","reason":"구성품 불명확 + 사후 가격 변경","triggered_message":"뭐가 오는지는 받아봐야 알죠"}

대화:
[판매자] 지금 1자리 남았어요. 10분 안에 결제 안 하시면 취소예요.
출력:
{"risk_level":"WARNING","reason":"과도한 마감 압박 — 판단력 저하 유도","triggered_message":"10분 안에 결제 안 하시면 취소예요"}

대화:
[판매자] 지금 바로 안 보내시면 다른 분한테 넘어가요. 이름 안 바꾸면 돈 날릴 수도 있어요. 빨리 입금해주세요.
출력:
{"risk_level":"DANGER","reason":"협박성 입금 압박 — 금전 손실 위협으로 빠른 송금 유도","triggered_message":"이름 안 바꾸면 돈 날릴 수도 있어요. 빨리 입금해주세요"}

대화:
[구매자] 배송비 포함 총 얼마예요?
[판매자] 28,000원 + 배송비 3,000원이에요. 플랫폼 결제창으로 안내드릴게요.
출력:
{"risk_level":"SAFE","reason":"정상 거래 — 플랫폼 결제 안내","triggered_message":null}

[반환 형식] JSON만, 설명 없이:
{"risk_level":"SAFE|WARNING|DANGER","reason":"이유","triggered_message":"해당 메시지 또는 null"}"""


@dataclass
class LLMResult:
    risk_level: str
    reason: str
    triggered_message: str | None


def _build_conversation(messages: list[dict]) -> str:
    window = messages[-WINDOW_SIZE:]
    return "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in window)


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
    user_content = f"[대화]\n{_build_conversation(messages)}"
    try:
        res = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        res.raise_for_status()
        return _parse(res.json().get("message", {}).get("content", ""))
    except httpx.TimeoutException:
        return LLMResult(risk_level="SAFE", reason="LLM 타임아웃", triggered_message=None)
    except Exception as e:
        return LLMResult(risk_level="SAFE", reason=f"LLM 오류: {e}", triggered_message=None)
