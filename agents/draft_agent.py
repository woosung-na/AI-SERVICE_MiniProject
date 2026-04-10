"""
Draft Agent (The Writer)
- 입력: analysis_json ONLY (원본 문서 접근 금지 → 할루시네이션 억제)
- 역할: JSON 데이터를 보고서 목차에 맞게 마크다운으로 서술
- 구현: gpt-4.1 (최종 품질 우선)
"""

import os
import time
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState
from metrics import MetricsTracker
from prompts.draft_prompt import DRAFT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def draft_agent_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Draft Agent 노드: analysis_json → 마크다운 보고서 초안"""
    t0 = time.time()
    logger.info("[Draft Agent] 보고서 초안 작성 시작...")

    analysis_json = state.get("analysis_json", {})
    query = state.get("query", "")
    iteration_count = state.get("iteration_count", 0)
    is_fallback = iteration_count >= 3

    # Fallback 시 "정보 부족" 명시 지시 포함
    fallback_note = (
        "\n\n⚠️ 주의: 데이터 수집이 충분하지 않습니다. "
        "보고서에 '정보 부족으로 인한 추정' 섹션을 명시하세요."
        if is_fallback else ""
    )

    llm = ChatOpenAI(
        model=os.getenv("DRAFT_MODEL", "gpt-4.1"),
        temperature=0.3,  # 약간의 서술 다양성
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", DRAFT_SYSTEM_PROMPT),
        ("human", (
            "원본 질의: {query}\n\n"
            "분석 JSON 데이터:\n{analysis_json}\n\n"
            "작성 날짜: {date}{fallback_note}\n\n"
            "위 데이터만을 기반으로 보고서를 작성하세요. "
            "JSON에 없는 정보는 절대 추가하지 마세요."
        )),
    ])

    try:
        import json
        response = (prompt | llm).invoke({
            "query": query,
            "analysis_json": json.dumps(analysis_json, ensure_ascii=False, indent=2),
            "date": datetime.now().strftime("%Y년 %m월 %d일"),
            "fallback_note": fallback_note,
        })
        draft_report = response.content
        logger.info(f"[Draft Agent] 초안 작성 완료 ({len(draft_report)}자)")
    except Exception as e:
        logger.error(f"[Draft Agent] 실패: {e}")
        draft_report = _fallback_draft(analysis_json, query, is_fallback)

    elapsed = time.time() - t0
    if metrics:
        metrics.record("draft_agent", {
            "report_length": len(draft_report),
            "is_fallback": is_fallback,
            "elapsed_sec": round(elapsed, 2),
        })

    return {**state, "draft_report": draft_report}


def _fallback_draft(analysis_json: dict, query: str, is_fallback: bool) -> str:
    """LLM 실패 시 JSON 직접 마크다운 변환.

    competitors는 List[dict](company_name 키) 또는 레거시 dict 양쪽 처리.
    """
    lines = [
        "# HBM4/PIM/CXL 기술 전략 분석 보고서",
        f"\n> 질의: {query}",
        "\n---\n",
        "## ⚠️ 분석 주의사항" if is_fallback else "",
        "> 데이터 부족으로 일부 항목은 간접 지표 기반 추정입니다.\n" if is_fallback else "",
        "## 경쟁사별 TRL 분석\n",
    ]

    competitors_raw = analysis_json.get("competitors", [])
    if isinstance(competitors_raw, dict):
        items = list(competitors_raw.items())
    else:
        items = [(c.get("company_name", f"comp_{i}"), c) for i, c in enumerate(competitors_raw)]

    for comp, data in items:
        lines.append(f"### {comp}")
        lines.append(f"- **TRL**: {data.get('trl', 'N/A')}")
        lines.append(f"- **위협 수준**: {data.get('threat_level', 'N/A')}")
        for ev in data.get("trl_evidence", []):
            lines.append(f"  - {ev}")
    return "\n".join(lines)
