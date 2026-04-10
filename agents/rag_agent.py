"""
RAG Agent
- FAISS + BM25 Ensemble Retrieval
- CacheBackedEmbeddings (text-embedding-3-small)
- data/ 디렉토리의 PDF 문서 검색
- 이중 쿼리 전략: 긍정 증거 + 반론/리스크 문서 강제 포함 (확증 편향 방지)
"""

import os
import glob
import hashlib
import time
import logging
from pathlib import Path
from state import ResearchState
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
_retriever_cache = {}  # 재사용: 동일 문서셋 재인덱싱 방지

# 이중 쿼리 설정
_PRO_DOC_LIMIT = 5    # 긍정 증거 상위 N개
_RISK_DOC_LIMIT = 3   # 반론 문서 상위 N개


def rag_agent_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """RAG Agent 노드: 이중 쿼리 Ensemble Retrieval (긍정 + 반론)"""
    t0 = time.time()

    query = state.get("revised_query") or state.get("query", "")
    logger.info(f"[RAG Agent] 쿼리: {query[:80]}...")

    retriever = _get_or_build_retriever()
    if retriever is None:
        logger.warning("[RAG Agent] 리트리버 없음. data/에 PDF를 추가하세요.")
        elapsed = time.time() - t0
        if metrics:
            metrics.record("rag_agent", {"status": "no_data", "elapsed_sec": round(elapsed, 2)})
        return {**state, "rag_docs": []}

    try:
        # ── 긍정 증거 쿼리 (원본) ──────────────────────────────
        pro_docs = retriever.invoke(query)

        # ── 반론/리스크 쿼리 (반론 문서 강제 포함) ─────────────
        counter_query = _build_counter_query(query)
        risk_docs = retriever.invoke(counter_query)
        logger.info(
            f"[RAG Agent] 긍정={len(pro_docs)}개, 반론쿼리={len(risk_docs)}개"
        )

        # ── 병합 및 중복 제거 ──────────────────────────────────
        merged = _merge_dedup(pro_docs[:_PRO_DOC_LIMIT], risk_docs[:_RISK_DOC_LIMIT])
        doc_texts = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in merged
        ]

        # source_type별 분포 로깅
        type_dist = {}
        for d in doc_texts:
            st = d["metadata"].get("source_type", "unknown")
            type_dist[st] = type_dist.get(st, 0) + 1
        logger.info(f"[RAG Agent] 최종 {len(doc_texts)}개 | 분포: {type_dist}")

    except Exception as e:
        logger.error(f"[RAG Agent] 검색 실패: {e}")
        doc_texts = []
        type_dist = {}

    elapsed = time.time() - t0
    if metrics:
        metrics.record("rag_agent", {
            "query": query[:60],
            "doc_count": len(doc_texts),
            "source_type_dist": type_dist,
            "elapsed_sec": round(elapsed, 2),
        })

    return {**state, "rag_docs": doc_texts, "iteration_count": state.get("iteration_count", 0) + 1}


def _build_counter_query(query: str) -> str:
    """반론/리스크 문서를 끌어올리는 보완 쿼리 생성"""
    risk_terms = "yield risk challenges JEDEC limitations NFI barriers delayed commercialization failure"
    return f"{query} {risk_terms}"


def _merge_dedup(pro_docs: list, risk_docs: list) -> list:
    """
    긍정 문서 + 반론 문서 병합 (content 해시 기반 중복 제거)
    반론 문서는 리스트 후반부 삽입 → _build_context()에서 섹션 분리 시 활용
    """
    seen = set()
    result = []

    def _add(docs):
        for doc in docs:
            key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                result.append(doc)

    _add(pro_docs)   # 긍정 증거 먼저
    _add(risk_docs)  # 반론 이후 (중복이면 건너뜀)
    return result


def _get_or_build_retriever():
    """PDF 파일 목록 기반으로 Retriever 캐싱"""
    from rag.pdf import PDFRetrievalChain

    pdf_files = sorted(glob.glob(str(DATA_DIR / "*.pdf")))
    if not pdf_files:
        return None

    cache_key = "_".join(pdf_files)
    if cache_key in _retriever_cache:
        logger.info("[RAG Agent] 캐시된 Retriever 사용")
        return _retriever_cache[cache_key]

    logger.info(f"[RAG Agent] PDF {len(pdf_files)}개 인덱싱 시작...")
    chain = PDFRetrievalChain(source_uri=pdf_files)
    chain.create_chain()
    _retriever_cache[cache_key] = chain.retriever
    logger.info("[RAG Agent] 인덱싱 완료")
    return chain.retriever
