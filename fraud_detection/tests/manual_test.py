"""
직접 채팅 입력해서 사기 탐지 테스트
$ python tests/manual_test.py
"""

import httpx

BASE_URL = "http://localhost:8003/api/fraud/analyze"
history: list[dict] = []


def send(role: str, content: str) -> dict:
    history.append({"sender_id": "u1", "role": role, "content": content, "timestamp": "2024-01-01T10:00:00"})
    res = httpx.post(BASE_URL, json={"room_id": "manual-test", "messages": history}, timeout=60)
    return res.json()


def print_result(result: dict):
    action = result["action"]
    level  = result["llm_risk_level"]
    reason = result["llm_reason"]
    matched = result["matched_patterns"]

    if action == "show_danger_banner":
        label = "🚨 DANGER"
    elif action == "show_warning_banner":
        label = "⚠️  WARNING"
    else:
        label = "✅ SAFE"

    print(f"\n{label}  |  LLM: {level}")
    if matched:
        print(f"규칙 탐지: {', '.join(matched)}")
    print(f"이유: {reason}")
    print("-" * 50)


def main():
    print("=" * 50)
    print("분철 사기 탐지 테스트")
    print("role 입력: 1 = 판매자, 2 = 구매자")
    print("'초기화' 입력 시 대화 내역 리셋")
    print("'종료' 입력 시 종료")
    print("=" * 50)

    while True:
        role_input = input("\nrole (1/2): ").strip()
        if role_input == "종료":
            break
        if role_input == "초기화":
            history.clear()
            print("대화 내역이 초기화됐습니다.")
            continue

        role = "판매자" if role_input == "1" else "구매자"
        content = input(f"[{role}] 메시지: ").strip()
        if not content:
            continue

        print("분석 중...")
        result = send(role, content)
        print_result(result)


if __name__ == "__main__":
    main()
