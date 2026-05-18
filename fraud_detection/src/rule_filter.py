"""
1단계: 규칙 기반 필터
즉각적인 DANGER 패턴을 정규식으로 탐지한다.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RuleResult:
    triggered: bool
    matched_patterns: list[str] = field(default_factory=list)


PATTERNS: list[tuple[str, str]] = [
    (
        "선입금 유도",
        r"(먼저\s*입금|입금\s*먼저|선입금|입금\s*후\s*발송|보내\s*주시면\s*바로|계좌\s*로\s*먼저)",
    ),
    (
        "외부 결제 수단 유도",
        r"(카카오\s*페이|문화\s*상품권|구글\s*플레이|틱톡\s*기프트|상품권\s*으로|페이\s*로\s*보내)",
    ),
    (
        "외부 채널 이동 유도",
        r"(오픈\s*채팅|카톡\s*ID|카카오톡\s*으로|개인\s*톡|다른\s*데서|따로\s*연락)",
    ),
    (
        "안전결제 사칭",
        r"(안전\s*결제\s*링크|안전\s*거래\s*링크|에스크로\s*링크|결제\s*링크\s*로)",
    ),
    (
        "외부 URL 포함",
        r"(https?://|bit\.ly|tinyurl|링크\s*클릭|주소\s*입력)",
    ),
    (
        "개인정보 직접 요청",
        r"(계좌\s*번호\s*랑|이름\s*이랑\s*계좌|주민\s*번호|신분증|통장\s*사본)",
    ),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in PATTERNS]


def check(messages: list[str]) -> RuleResult:
    """메시지 목록에서 위험 패턴을 탐지한다."""
    text = " ".join(messages)
    matched = [name for name, regex in _COMPILED if regex.search(text)]
    return RuleResult(triggered=bool(matched), matched_patterns=matched)
