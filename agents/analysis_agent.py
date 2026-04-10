"""
Analysis Agent (The Brain)
- 입력: rag_docs + web_results
- 역할: TRL 단계 판정 + 경쟁사별 위협 수준 → 구조화 JSON 출력
- 구현: structured output (with_structured_output, strict=False) + gpt-4.1-mini
- CompetitorsMap: Samsung·Micron 고정 필드 → 누락 방지
- supporting_quotes 필드 필수 → Draft Agent의 Context Blindness 방지
"""

import os
import time
import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from state import ResearchState
from metrics import MetricsTracker
from prompts.analysis_prompt import ANALYSIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── 출력 스키마 ───────────────────────────────────────────────

class CompetitorAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trl: int = Field(ge=1, le=9, description="TRL 단계 (1-9)")
    trl_evidence: List[str] = Field(description="TRL 판정 근거 목록")
    threat_level: str = Field(description="위협 수준: high/medium/low")
    supporting_quotes: List[str] = Field(
        description="원문 인용구 (Draft Agent Context 보강용)"
    )
    key_activities: List[str] = Field(description="주요 R&D 활동")
    timeline_estimate: Optional[str] = Field(None, description="상용화 예상 시점")


class TechnologyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_state: str = Field(description="기술 현황 요약")
    key_challenges: List[str] = Field(description="핵심 기술 과제")
    market_readiness: str = Field(description="시장 준비도: emerging/developing/maturing")


class CompetitorsMap(BaseModel):
    """경쟁사별 분析 — Samsung·Micron 고정 필드로 누락 방지"""
    model_config = ConfigDict(extra="forbid")
    Samsung: CompetitorAnalysis = Field(description="Samsung 분析")
    Micron: CompetitorAnalysis = Field(description="Micron 분析")


class TechnologiesMap(BaseModel):
    """기술별 현황 — HBM4·PIM·CXL 고정 필드"""
    model_config = ConfigDict(extra="forbid")
    HBM4: TechnologyStatus = Field(description="HBM4 현황")
    PIM: TechnologyStatus = Field(description="PIM 현황")
    CXL: TechnologyStatus = Field(description="CXL 현황")


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    competitors: CompetitorsMap = Field(description="경쟁사별 분析 (Samsung·Micron 필수)")
    technologies: TechnologiesMap = Field(description="기술별 현황 (HBM4·PIM·CXL 필수)")
    overall_threat_summary: str = Field(description="종합 위협 요약 (2-3문장)")
    data_quality_note: str = Field(description="데이터 신뢰도 및 한계 명시 (리스크 교차검증 결과 포함)")


def analysis_agent_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Analysis Agent 노드: RAG + Web 결과 → 구조화 JSON"""
    t0 = time.time()
    logger.info("[Analysis Agent] 분析 시작...")

    rag_docs = state.get("rag_docs", [])
    web_results = state.get("web_results", [])
    feedback = state.get("feedback", "")

    context = _build_context(rag_docs, web_results)

    llm = ChatOpenAI(
        model=os.getenv("ANALYSIS_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )
    # strict=False: Optional 필드(timeline_estimate) 지원 + CompetitorsMap 고정 스키마
    structured_llm = llm.with_structured_output(AnalysisOutput, strict=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYSIS_SYSTEM_PROMPT),
        ("human", "다음 수집 데이터를 분析하여 구조화된 JSON을 생성하세요.\n\n{context}\n\n이전 피드백: {feedback}"),
    ])

    chain = prompt | structured_llm

    try:
        result: AnalysisOutput = chain.invoke({
            "context": context,
            "feedback": feedback or "없음",
        })
        analysis_json = result.model_dump()
        competitor_keys = list(analysis_json.get("competitors", {}).keys())
        logger.info(f"[Analysis Agent] 완료: 경쟁사={competitor_keys}")
    except Exception as e:
        logger.error(f"[Analysis Agent] 실패: {e}")
        analysis_json = _fallback_analysis(feedback)

    elapsed = time.time() - t0
    if metrics:
        # competitors는 dict 형식: {"Samsung": {...}, "Micron": {...}}
        competitor_trls = {
            k: v.get("trl") for k, v in analysis_json.get("competitors", {}).items()
        }
        obj_score = metrics.compute_objectivity_score(rag_docs)
        metrics.record("objectivity", obj_score)
        metrics.record("analysis_agent", {
            "competitor_trls": competitor_trls,
            "elapsed_sec": round(elapsed, 2),
            "rag_doc_count": len(rag_docs),
            "web_result_count": len(web_results),
            "objectivity_passed": obj_score.get("passed"),
            "risk_ratio": obj_score.get("risk_ratio", 0.0),
        })

    return {
        **state,
        "analysis_json": analysis_json,
        "trl_passed": False,  # TRL Judge가 이후 판정
    }


def _build_context(rag_docs: list, web_results: list) -> str:
    """
    RAG + Web 결과를 LLM 입력 컨텍스트로 변환.
    source_type 메타데이터 기반으로 2섹션 분리:
      - 제조사 주장 및 기술 현황 (ir_press / academic / foundry / market)
      - ⚠️ 리스크 및 반론 자료 (risk / industry) ← 교차검증 필수
    메타데이터 없는 문서(PDF-less 모드)는 첫 번째 섹션으로 fallback.
    """
    RISK_TYPES = {"risk", "industry"}
    pro_docs, risk_docs_rag = [], []

    for doc in rag_docs[:10]:
        if not isinstance(doc, dict):
            pro_docs.append(doc)
            continue
        stype = doc.get("metadata", {}).get("source_type", "general")
        if stype in RISK_TYPES:
            risk_docs_rag.append(doc)
        else:
            pro_docs.append(doc)

    lines = ["## 제조사 주장 및 기술 현황 (IR / 학술 / 파운드리)"]
    if pro_docs:
        for i, doc in enumerate(pro_docs, 1):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            scat = doc.get("metadata", {}).get("source_category", "") if isinstance(doc, dict) else ""
            sfile = doc.get("metadata", {}).get("source_file", "") if isinstance(doc, dict) else ""
            tag = f"[{scat}|{sfile}]" if scat else ""
            lines.append(f"[문서 {i}] {tag} {content[:600]}")
    else:
        lines.append("(해당 문서 없음)")

    lines.append("\n## ⚠️ 리스크 및 반론 자료 (교차검증 필수 — 제조사 주장과 대조할 것)")
    if risk_docs_rag:
        for i, doc in enumerate(risk_docs_rag, 1):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            scat = doc.get("metadata", {}).get("source_category", "")
            sfile = doc.get("metadata", {}).get("source_file", "")
            lines.append(f"[리스크 문서 {i}] [{scat}|{sfile}] {content[:600]}")
    else:
        lines.append("(리스크 문서 없음 — data/ 디렉토리에 반론 PDF 추가 권장)")

    lines.append("\n## 웹 검색 결과 (실시간 동향)")
    for i, r in enumerate(web_results[:15], 1):
        if isinstance(r, dict):
            title = r.get("title", "")
            content = r.get("content", "")[:400]
            url = r.get("url", "")
            lines.append(f"[웹 {i}] {title}\n{content}\n출처: {url}")
        else:
            lines.append(f"[웹 {i}] {str(r)[:400]}")

    return "\n\n".join(lines)


def _fallback_analysis(feedback: str) -> dict:
    """LLM 실패 시 최소 구조 반환 (CompetitorsMap 딕셔너리 형식)"""
    return {
        "competitors": {
            "Samsung": {
                "trl": 5, "trl_evidence": ["데이터 부족"],
                "threat_level": "medium", "supporting_quotes": [],
                "key_activities": [], "timeline_estimate": None,
            },
            "Micron": {
                "trl": 5, "trl_evidence": ["데이터 부족"],
                "threat_level": "medium", "supporting_quotes": [],
                "key_activities": [], "timeline_estimate": None,
            },
        },
        "technologies": {
            "HBM4": {"current_state": "분석 실패", "key_challenges": [], "market_readiness": "developing"},
            "PIM": {"current_state": "분析 실패", "key_challenges": [], "market_readiness": "developing"},
            "CXL": {"current_state": "분析 실패", "key_challenges": [], "market_readiness": "developing"},
        },
        "overall_threat_summary": "분析 실패로 인한 폴백 데이터.",
        "data_quality_note": f"LLM 分析 실패. 피드백: {feedback}",
    }
