"""
ResearchState: LangGraph 전체 공유 State 정의
설계서 C. Agent 정의 → State 설계 준수
"""

from typing import TypedDict, Optional


class ResearchState(TypedDict):
    # ── 1. 입력 및 검색 관련 ──────────────────────────────
    query: str                      # 원본 질의
    revised_query: str              # Retrieval Judge가 재구성한 검색 쿼리
    rag_docs: list                  # RAG Agent 원본 문서 청크 (Agent 전용)
    web_results: list               # Web Agent 웹 검색 결과 (Agent 전용)

    # ── 2. Supervisor 전용 요약 (토큰 최적화) ─────────────
    collection_summary: str         # RAG + Web 통합 요약 → Supervisor 보고용

    # ── 3. 분석 및 검증 데이터 ────────────────────────────
    analysis_json: dict             # Analysis Agent 구조화 JSON (supporting_quotes 포함)

    # ── 4. 제어 신호 ──────────────────────────────────────
    retrieval_passed: bool          # Retrieval Judge 통과 여부
    trl_passed: bool                # TRL Judge 통과 여부
    feedback: str                   # Judge → Supervisor 반려 사유

    # ── 5. 출력물 ─────────────────────────────────────────
    draft_report: str               # Draft Agent 마크다운 초안
    final_report: str               # Formatting Node 최종 보고서

    # ── 6. 시스템 관리 ────────────────────────────────────
    iteration_count: int            # 루프 카운터 (Max=3, Fallback 트리거)
    next_action: str                # Supervisor → 다음 노드 결정


def initial_state(query: str) -> ResearchState:
    """초기 State 생성 헬퍼"""
    return ResearchState(
        query=query,
        revised_query="",
        rag_docs=[],
        web_results=[],
        collection_summary="",
        analysis_json={},
        retrieval_passed=False,
        trl_passed=False,
        feedback="",
        draft_report="",
        final_report="",
        iteration_count=0,
        next_action="rag_search",
    )
