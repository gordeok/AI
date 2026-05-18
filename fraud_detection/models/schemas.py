from pydantic import BaseModel


class Message(BaseModel):
    sender_id: str
    role: str       # 판매자 | 구매자
    content: str
    timestamp: str


class AnalyzeRequest(BaseModel):
    room_id: str
    messages: list[Message]


class AnalyzeResponse(BaseModel):
    rule_triggered: bool
    matched_patterns: list[str]
    llm_risk_level: str         # SAFE | WARNING | DANGER
    llm_reason: str
    action: str                 # none | show_warning_banner | show_danger_banner
