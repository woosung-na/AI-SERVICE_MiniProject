"""
Judges 모듈
1. Retrieval Judge: LLM 기반 관련성 점수 (Self-RAG 패턴)
2. TRL Judge: 정규식 기반 Python 함수 (LLM 호출 없음, 빠름)
"""

import os
import re
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from state import ResearchState
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

# ── TRL Judge: 강력 근거 판정 패턴 ─────────────────────────
STRONG_PATTERN = re.compile(
    r'\d+[\.\d]*\s*(TB/s|GB/s|GB|nm|%|layer|단|개|stack|die)\b'
    r'|Samsung|Micron|TSMC|BESI|AMAT|Hynix'
    r'|Hybrid\s*Bonding|HBM4|TSV|CXL\s*\d|PIM|AiMX'
    r'|ISSCC\s*\d{4}|Hot\s*Chips|IEEE',
    re.IGNORECASE,
)

RELEVANCE_THRESHOLD = 0.6  # Retrieval Judge 임계값


# ── 1. Retrieval Judge ────────────────────────────────────────

class RelevanceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="관련성 점수 (0.0~1.0)")
    reasoning: str = Field(description="판정 근거 (1-2문장)")
    revised_query: str = Field(description="미달 시 재구성 쿼리 (통과 시 원본 그대로)")


_RETRIEVAL_JUDGE_SYSTEM = """당신은 반도체 기술 문서 검색 품질 평가 전문가입니다.
수집된 문서들이 질의와 기술적으로 연관이 있는지 평가하세요. 

평가 기준:
- HBM4, PIM, CXL, Samsung, Micron 관련 구체적 기술/동향 포함 여부
- 관련성이 높으면 score ≥ 0.8
- 부분적 관련이면 score 0.6~0.8
- 관련성 낮으면 score < 0.6 → 쿼리 재구성 필요

score < {threshold} 시 revised_query에 더 구체적인 재구성 쿼리를 작성하세요.
예: "Samsung HBM4 2025 bandwidth specification ISSCC"
"""


def retrieval_judge_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Retrieval Judge 노드: 관련성 점수 기반 통과/재검색 판정"""
    t0 = time.time()
    logger.info("[Retrieval Judge] 관련성 평가 시작...")

    query = state.get("revised_query") or state.get("query", "")
    rag_docs = state.get("rag_docs", [])
    web_results = state.get("web_results", [])

    # 평가용 요약 컨텍스트 (토큰 절감)
    sample_docs = "\n---\n".join([
        d.get("content", str(d))[:300] for d in rag_docs[:5]
    ])
    sample_web = "\n---\n".join([
        (r.get("title", "") + ": " + r.get("content", "")[:200])
        if isinstance(r, dict) else str(r)[:200]
        for r in web_results[:5]
    ])

    llm = ChatOpenAI(
        model=os.getenv("JUDGE_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(RelevanceScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _RETRIEVAL_JUDGE_SYSTEM.format(threshold=RELEVANCE_THRESHOLD)),
        ("human", (
            "질의: {query}\n\n"
            "RAG 문서 샘플:\n{rag_docs}\n\n"
            "웹 결과 샘플:\n{web_results}\n\n"
            "관련성을 평가하고 score와 revised_query를 반환하세요."
        )),
    ])

    try:
        result: RelevanceScore = (prompt | structured_llm).invoke({
            "query": query,
            "rag_docs": sample_docs or "없음",
            "web_results": sample_web or "없음",
        })
        passed = result.score >= RELEVANCE_THRESHOLD
        revised_query = result.revised_query if not passed else query
        feedback = "" if passed else f"관련성 점수 {result.score:.2f} < {RELEVANCE_THRESHOLD}: {result.reasoning}"
        logger.info(f"[Retrieval Judge] score={result.score:.2f}, passed={passed}")
    except Exception as e:
        logger.error(f"[Retrieval Judge] LLM 오류: {e}. 통과로 처리.")
        passed, revised_query, feedback = True, query, ""
        result = type("R", (), {"score": 0.7, "reasoning": "LLM 오류"})()

    # 컬렉션 요약 생성 (Supervisor 토큰 절감용)
    collection_summary = _build_collection_summary(rag_docs, web_results, result.score if hasattr(result, 'score') else 0.7)

    elapsed = time.time() - t0
    if metrics:
        metrics.record("retrieval_judge", {
            "score": getattr(result, "score", 0.7),
            "passed": passed,
            "threshold": RELEVANCE_THRESHOLD,
            "elapsed_sec": round(elapsed, 2),
        })

    return {
        **state,
        "retrieval_passed": passed,
        "revised_query": revised_query,
        "feedback": feedback,
        "collection_summary": collection_summary,
    }


def _build_collection_summary(rag_docs: list, web_results: list, score: float) -> str:
    return (
        f"수집 현황: RAG {len(rag_docs)}개 청크, 웹 {len(web_results)}개 결과. "
        f"관련성 점수: {score:.2f}. "
        f"주요 출처: {', '.join(set([r.get('url','')[:30] for r in web_results[:3] if isinstance(r,dict)]))}..."
    )


# ── 2. TRL Judge ─────────────────────────────────────────────

def trl_judge_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """TRL Judge 노드: 정규식 기반 근거 충실도 검증 (LLM 호출 없음)"""
    t0 = time.time()
    logger.info("[TRL Judge] 근거 충실도 검증 시작...")

    analysis_json = state.get("analysis_json", {})
    passed, feedback, detail = _trl_judge_logic(analysis_json)

    elapsed = time.time() - t0
    logger.info(f"[TRL Judge] passed={passed}, detail={detail}")

    if metrics:
        metrics.record("trl_judge", {
            "passed": passed,
            "detail": detail,
            "elapsed_sec": round(elapsed, 3),
        })

    return {
        **state,
        "trl_passed": passed,
        "feedback": feedback,
    }


def _trl_judge_logic(analysis_json: dict) -> tuple[bool, str, dict]:
    """
    통과 조건 (OR):
    - 강력 근거 ≥ 1개 (수치+단위, 기업명, 기술 고유명사)
    - 일반 근거 ≥ 2개

    competitors는 List[dict] 형식 (company_name 키 보유) 또는
    레거시 dict 형식 모두 처리.
    """
    competitors_raw = analysis_json.get("competitors", [])
    if not competitors_raw:
        return False, "경쟁사 분석 데이터 없음", {}

    # 리스트(정상 경로) / 딕셔너리(레거시) 양쪽 정규화
    if isinstance(competitors_raw, dict):
        competitors_items = list(competitors_raw.items())
    else:
        competitors_items = [
            (c.get("company_name", f"comp_{i}"), c)
            for i, c in enumerate(competitors_raw)
        ]

    detail = {}
    for comp, data in competitors_items:
        trl = data.get("trl", 0)
        evidence = data.get("trl_evidence", [])
        quotes = data.get("supporting_quotes", [])
        all_evidence = evidence + quotes

        # TRL 4~6 구간만 엄격 검증 (그 외는 통과)
        if not (4 <= trl <= 6):
            detail[comp] = {"trl": trl, "check": "skip (TRL out of 4-6 range)", "passed": True}
            continue

        strong_count = sum(1 for e in all_evidence if STRONG_PATTERN.search(e))
        total_count = len(all_evidence)

        comp_passed = strong_count >= 1 or total_count >= 2
        detail[comp] = {
            "trl": trl,
            "strong_evidence": strong_count,
            "total_evidence": total_count,
            "passed": comp_passed,
        }

        if not comp_passed:
            feedback = (
                f"{comp}: TRL {trl} 근거 불충분 "
                f"(강력 근거 {strong_count}개, 전체 {total_count}개). "
                f"수치/기업명/기술명 포함된 구체적 근거 추가 필요."
            )
            return False, feedback, detail

    return True, "TRL 근거 검증 통과", detail
