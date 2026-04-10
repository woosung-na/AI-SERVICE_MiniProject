"""
Web Agent
- Tavily Search: 실시간 뉴스·채용공고·특허 동향 수집
- 확증 편향 방지: 긍정 쿼리 + 반론 쿼리 병렬 실행
  예: "Samsung HBM4 progress" + "Samsung HBM4 challenges"
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from state import ResearchState
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

_POSITIVE_SUFFIX = ["latest progress", "breakthrough", "achievement", "roadmap"]
_NEGATIVE_SUFFIX = ["challenges", "delays", "limitations", "competition risks"]

TARGETS = ["Samsung", "Micron"]
TECHS = ["HBM4", "PIM", "CXL"]


def web_agent_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Web Agent 노드: Tavily 실시간 웹 검색 (긍정+반론 쿼리)"""
    t0 = time.time()
    query = state.get("revised_query") or state.get("query", "")
    logger.info(f"[Web Agent] 기본 쿼리: {query[:80]}...")

    queries = _build_queries(query)
    results = _search_parallel(queries)

    elapsed = time.time() - t0
    logger.info(f"[Web Agent] 수집 완료: {len(results)}개 결과 ({elapsed:.1f}s)")

    if metrics:
        metrics.record("web_agent", {
            "query_count": len(queries),
            "result_count": len(results),
            "elapsed_sec": round(elapsed, 2),
        })

    return {**state, "web_results": results}


def _build_queries(base_query: str) -> list[str]:
    """긍정 + 반론 쿼리 생성으로 확증 편향 방지.

    부정(리스크) 쿼리를 먼저 배치하여 12개 절단 시에도
    모든 타겟·기술 조합의 반론 쿼리가 보장되도록 함.
    이전 구현은 Samsung 긍정 쿼리가 먼저 채워져 Micron 리스크 쿼리가
    전부 잘려나가는 구조적 편향이 있었음.
    """
    neg_queries = []
    pos_queries = []
    for target in TARGETS:
        for tech in TECHS:
            neg_queries.append(f"{target} {tech} {_NEGATIVE_SUFFIX[0]} 2025")
            for suffix in _POSITIVE_SUFFIX[:2]:
                pos_queries.append(f"{target} {tech} {suffix} 2024 2025")

    # base(1) + neg(6) + pos(5) = 12 — neg 전부 포함 후 pos로 채움
    return ([base_query] + neg_queries + pos_queries)[:12]


def _search_parallel(queries: list[str]) -> list[dict]:
    """Tavily 병렬 검색"""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily = TavilySearchResults(max_results=3)
    except Exception as e:
        logger.error(f"[Web Agent] Tavily 초기화 실패: {e}")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_query = {
            executor.submit(_single_search, tavily, q): q
            for q in queries
        }
        for future in as_completed(future_to_query):
            q = future_to_query[future]
            try:
                items = future.result()
                results.extend(items)
                logger.debug(f"[Web Agent] '{q[:40]}' → {len(items)}개")
            except Exception as e:
                logger.warning(f"[Web Agent] 검색 실패 '{q[:40]}': {e}")

    # 중복 URL 제거
    seen_urls = set()
    deduped = []
    for r in results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)
    return deduped


def _single_search(tavily, query: str) -> list[dict]:
    raw = tavily.invoke(query)
    return raw if isinstance(raw, list) else []
