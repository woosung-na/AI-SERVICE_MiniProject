"""
Supervisor Agent
- 상태 기반 결정론적 라우터: next_action 결정
- 설계서: Supervisor는 노드이며 매 단계마다 다음 지시를 내리는 주체
- 가능한 next_action: rag_search | web_search | retrieval_judge | analysis |
                       trl_judge | draft | format | end

[설계 변경 이유]
LLM 기반 라우팅은 rag_docs+web_results가 모두 있는데도
retrieval_judge 대신 rag_search를 반복 선택하는 잘못된 결정을 내렸음.
이 파이프라인의 전이 조건은 State 플래그에 의해 완전히 결정론적이므로
LLM을 제거하고 State 기반 규칙 라우팅으로 교체.
속도 향상(LLM 호출 제거) + 무결한 흐름 보장이 부가 이점.
"""

import time
import logging
from state import ResearchState
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def supervisor_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Supervisor 노드: 상태 기반 결정론적 라우팅"""
    t0 = time.time()
    iteration = state.get("iteration_count", 0)

    next_action = _route(state, iteration)

    elapsed = time.time() - t0
    logger.info(f"[Supervisor] next_action={next_action} | iteration={iteration}")
    if metrics:
        metrics.record("supervisor", {
            "next_action": next_action,
            "iteration": iteration,
            "elapsed_sec": round(elapsed, 4),
        })

    return {**state, "next_action": next_action}


def _route(state: ResearchState, iteration: int) -> str:
    """
    우선순위 순으로 라우팅 결정.

    1. 최종 보고서 완성 → end          (format 이후 재진입 시 무한루프 방지)
    2. 초안 완성 → format               (fallback draft 루프 방지)
    3. 최대 반복 도달 → draft (fallback)
    4. 정상 흐름: State 플래그 순서대로 결정론적 라우팅
       rag_docs 없음          → rag_search
       web_results 없음       → web_search
       retrieval_passed=False → retrieval_judge
       analysis_json 없음     → analysis
       trl_passed=False       → trl_judge
       draft_report 없음      → draft
       final_report 없음      → format
       그 외                  → end
    """
    # ── 1. 종료 가드 ─────────────────────────────────────────
    if state.get("final_report"):
        logger.info("[Supervisor] final_report 완성 → end")
        return "end"

    # ── 2. 초안 가드 ─────────────────────────────────────────
    if state.get("draft_report"):
        logger.info("[Supervisor] draft_report 완성 → format")
        return "format"

    # ── 3. Fallback ──────────────────────────────────────────
    if iteration >= MAX_ITERATIONS:
        logger.warning(
            f"[Supervisor] Fallback 발동 (iteration={iteration} ≥ {MAX_ITERATIONS}). "
            "수집된 데이터로 draft 강제 진행."
        )
        return "draft"

    # ── 4. 정상 흐름 결정론적 라우팅 ─────────────────────────
    if not state.get("rag_docs"):
        return "rag_search"
    if not state.get("web_results"):
        return "web_search"
    if not state.get("retrieval_passed"):
        return "retrieval_judge"
    if not state.get("analysis_json"):
        return "analysis"
    if not state.get("trl_passed"):
        return "trl_judge"
    if not state.get("draft_report"):
        return "draft"
    if not state.get("final_report"):
        return "format"
    return "end"
