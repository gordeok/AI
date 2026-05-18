from fastapi import APIRouter
from models.schemas import AnalyzeRequest, AnalyzeResponse
from src import rule_filter, llm_analyzer

router = APIRouter(prefix="/api/fraud", tags=["fraud"])


def _determine_action(rule_triggered: bool, llm_risk_level: str) -> str:
    if rule_triggered or llm_risk_level == "DANGER":
        return "show_danger_banner"
    if llm_risk_level == "WARNING":
        return "show_warning_banner"
    return "none"


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    contents = [m.content for m in req.messages]
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # 1단계: 규칙 기반 필터
    rule_result = rule_filter.check(contents)

    # 규칙에 걸리면 LLM 생략하고 즉시 반환
    if rule_result.triggered:
        return AnalyzeResponse(
            rule_triggered=True,
            matched_patterns=rule_result.matched_patterns,
            llm_risk_level="DANGER",
            llm_reason="규칙 기반 필터 탐지",
            action="show_danger_banner",
        )

    # 2단계: EXAONE 맥락 분석
    llm_result = llm_analyzer.analyze(messages)

    return AnalyzeResponse(
        rule_triggered=False,
        matched_patterns=[],
        llm_risk_level=llm_result.risk_level,
        llm_reason=llm_result.reason,
        action=_determine_action(False, llm_result.risk_level),
    )
