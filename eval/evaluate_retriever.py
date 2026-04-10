"""
오프라인 Retriever 평가 스크립트
설계서 F. 성능 지표 및 검증

지표:
- Hit Rate@K (K=3, 5): 정답 문서가 상위 K개 안에 있는 비율
- MRR (Mean Reciprocal Rank): 첫 번째 정답 문서의 순위 역수 평균

사용법:
    python eval/evaluate_retriever.py --pdf data/paper1.pdf --k 3 5
    python eval/evaluate_retriever.py --testset eval/testset.json
"""

import os
import sys
import json
import argparse
import random
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 지표 계산 함수 ─────────────────────────────────────────

def compute_hit_rate(retrieved_ids: List[List[str]], relevant_ids: List[str], k: int) -> float:
    """
    Hit Rate@K: 각 쿼리에서 정답 문서가 상위 K개 안에 있으면 1, 아니면 0
    retrieved_ids: [[doc_id1, doc_id2, ...], ...] (쿼리별 검색 결과)
    relevant_ids: [relevant_doc_id, ...] (쿼리별 정답 문서 ID)
    """
    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k = retrieved[:k]
        if any(r in top_k for r in (relevant if isinstance(relevant, list) else [relevant])):
            hits += 1
    return hits / len(retrieved_ids) if retrieved_ids else 0.0


def compute_mrr(retrieved_ids: List[List[str]], relevant_ids: List[str]) -> float:
    """
    MRR: 각 쿼리에서 첫 번째 정답 문서 순위의 역수 평균
    """
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant if isinstance(relevant, list) else [relevant])
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in rel_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_precision_at_k(retrieved_ids: List[List[str]], relevant_ids: List[str], k: int) -> float:
    """Precision@K: 상위 K 중 정답 비율"""
    precisions = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant if isinstance(relevant, list) else [relevant])
        top_k = retrieved[:k]
        precision = len([d for d in top_k if d in rel_set]) / k
        precisions.append(precision)
    return float(np.mean(precisions)) if precisions else 0.0


# ── 테스트셋 생성 ─────────────────────────────────────────

def generate_testset_with_llm(split_docs: list, n_samples: int = 20) -> List[dict]:
    """
    LLM으로 질문-정답 쌍 생성 (참조: langchain-v1/14-Retriever/10-Retriever-Evaluation.ipynb)
    각 청크에서 질문 생성 → 해당 청크가 정답
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    prompt = PromptTemplate.from_template(
        "다음 반도체 기술 문서 내용을 기반으로, "
        "이 문서에서만 답할 수 있는 구체적인 기술 질문을 1개 생성하세요.\n\n"
        "문서:\n{context}\n\n"
        "질문만 출력하세요 (부연 설명 없이):"
    )

    random.seed(42)
    eligible = [d for d in split_docs if len(d.page_content.strip()) >= 100]
    sampled = random.sample(eligible, min(n_samples, len(eligible)))

    testset = []
    for doc in tqdm(sampled, desc="테스트셋 생성"):
        try:
            question = (prompt | llm).invoke({"context": doc.page_content[:800]}).content.strip()
            testset.append({
                "question": question,
                "relevant_chunk_id": doc.metadata.get("chunk_id", hash(doc.page_content)),
                "source": doc.metadata.get("source", ""),
                "ground_truth_content": doc.page_content[:200],
            })
        except Exception as e:
            logger.warning(f"질문 생성 실패: {e}")
    return testset


# ── 평가 실행 ─────────────────────────────────────────────

def evaluate(
    retriever,
    testset: List[dict],
    k_values: List[int] = [3, 5],
) -> dict:
    """Retriever 평가 실행"""
    logger.info(f"평가 시작: {len(testset)}개 질문, K={k_values}")

    all_retrieved_ids = []
    all_relevant_ids = []

    for item in tqdm(testset, desc="검색 평가"):
        question = item["question"]
        relevant_id = str(item["relevant_chunk_id"])

        try:
            docs = retriever.invoke(question)
            retrieved_ids = [
                str(d.metadata.get("chunk_id", hash(d.page_content)))
                for d in docs
            ]
        except Exception as e:
            logger.warning(f"검색 실패: {e}")
            retrieved_ids = []

        all_retrieved_ids.append(retrieved_ids)
        all_relevant_ids.append([relevant_id])

    results = {
        "n_questions": len(testset),
        "mrr": round(compute_mrr(all_retrieved_ids, all_relevant_ids), 4),
    }
    for k in k_values:
        results[f"hit_rate@{k}"] = round(compute_hit_rate(all_retrieved_ids, all_relevant_ids, k), 4)
        results[f"precision@{k}"] = round(compute_precision_at_k(all_retrieved_ids, all_relevant_ids, k), 4)

    return results


def print_results(results: dict, k_values: List[int] = [3, 5]):
    """평가 결과 출력"""
    sep = "=" * 50
    print(f"\n{sep}")
    print("  📊 Retriever 평가 결과")
    print(sep)
    print(f"  질문 수     : {results['n_questions']}개")
    print(f"  MRR         : {results['mrr']:.4f}  {'✅' if results['mrr'] >= 0.5 else '❌'} (기준: ≥ 0.5)")
    for k in k_values:
        hr = results.get(f"hit_rate@{k}", 0)
        target = 0.7 if k == 5 else 0.6
        print(f"  Hit Rate@{k} : {hr:.4f}  {'✅' if hr >= target else '❌'} (기준: ≥ {target})")
        print(f"  Precision@{k}: {results.get(f'precision@{k}', 0):.4f}")
    print(sep)

    # Objectivity Score 출력 (있을 경우)
    if "objectivity" in results:
        obj = results["objectivity"]
        passed = obj.get("passed")
        ratio = obj.get("risk_ratio", 0)
        status = "✅" if passed else ("⚠️" if passed is False else "⚪")
        print(f"\n🎯 Objectivity Score:")
        print(f"  반론 문서 비율  : {ratio:.1%} ({obj.get('risk_doc_count',0)}/{obj.get('total_queries',0)}개 쿼리)")
        print(f"  편향 없음 판정  : {status} {'통과' if passed else ('위험' if passed is False else 'N/A')}")
        if obj.get("pro_only_queries"):
            print(f"  반론 없는 쿼리  : {', '.join(obj['pro_only_queries'][:3])} ...")
        print()

    # 권장 α 값 힌트
    print(f"\n💡 α 조정 권장:")
    if results["mrr"] < 0.5 or results.get("hit_rate@5", 0) < 0.7:
        print("  현재 성능 기준 미달. α(BM25/Dense 가중치) 조정을 고려하세요.")
        print("  예: BM25를 높이려면 bm25_weight=0.7, dense_weight=0.3")
    else:
        print("  현재 α=0.5 설정으로 기준 달성. 유지 권장.")
    print()


# ── Objectivity Score 평가 ────────────────────────────────────

RISK_FILENAMES_DEFAULT = [
    "HBM4_Height_Standard_JEDEC_Risk",
    "AI_Energy_Demand_IEA_Report",
    "CXL_Adoption_Barriers_DigitalToday",
    "NFI_HybridBonding_Metrology_Whitepaper",
]

STANDARD_ANALYSIS_QUERIES = [
    "Samsung HBM4 TRL 2025 bandwidth specification",
    "SK Hynix HBM4 16-layer 48GB production",
    "Micron HBM4 12-layer bandwidth commercialization",
    "CXL 3.1 adoption Google Nvidia",
    "PIM AiMX power efficiency TCO",
    "TSMC CoWoS packaging HBM4 roadmap",
    "Hybrid Bonding HBM4 yield metrology",
]


def evaluate_objectivity(
    retriever,
    standard_queries: List[str] = None,
    risk_filenames: List[str] = None,
) -> dict:
    """
    Objectivity Score 오프라인 평가:
    표준 분석 쿼리 실행 → 반론/리스크 문서가 결과에 포함되는지 확인.

    Args:
        retriever: 평가할 Retriever
        standard_queries: 테스트할 쿼리 목록 (기본: STANDARD_ANALYSIS_QUERIES)
        risk_filenames: 리스크 문서로 간주할 파일명 패턴 목록

    Returns:
        {risk_coverage_rate, pro_only_queries, per_query_details}
    """
    queries = standard_queries or STANDARD_ANALYSIS_QUERIES
    risk_patterns = risk_filenames or RISK_FILENAMES_DEFAULT
    risk_types = {"risk", "industry"}

    per_query = []
    pro_only_queries = []

    for query in tqdm(queries, desc="Objectivity 평가"):
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            logger.warning(f"쿼리 실패 '{query[:40]}': {e}")
            per_query.append({"query": query, "risk_found": False, "risk_count": 0, "total": 0})
            pro_only_queries.append(query)
            continue

        # source_type 또는 파일명 패턴으로 리스크 문서 감지
        risk_docs_found = []
        for doc in docs:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            stype = meta.get("source_type", "")
            sfile = meta.get("source_file", "")
            is_risk = (
                stype in risk_types or
                any(p.lower() in sfile.lower() for p in risk_patterns)
            )
            if is_risk:
                risk_docs_found.append(sfile or stype)

        has_risk = len(risk_docs_found) > 0
        per_query.append({
            "query": query,
            "risk_found": has_risk,
            "risk_count": len(risk_docs_found),
            "risk_sources": list(set(risk_docs_found)),
            "total": len(docs),
        })
        if not has_risk:
            pro_only_queries.append(query)

    risk_coverage_rate = sum(1 for r in per_query if r["risk_found"]) / len(per_query) if per_query else 0.0

    return {
        "n_queries": len(queries),
        "risk_coverage_rate": round(risk_coverage_rate, 3),
        "pro_only_queries": pro_only_queries,
        "passed": risk_coverage_rate >= 0.5,  # 50% 이상 쿼리에서 반론 문서 포함
        "per_query_details": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Retriever 오프라인 평가 (Hit Rate@K, MRR, Objectivity)")
    parser.add_argument("--pdf", nargs="+", default=[], help="평가용 PDF 파일 경로")
    parser.add_argument("--testset", type=str, default=None, help="기존 테스트셋 JSON 경로")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5], help="Hit Rate K 값 목록")
    parser.add_argument("--n-samples", type=int, default=20, help="테스트셋 샘플 수")
    parser.add_argument("--alpha", type=float, default=0.5, help="BM25 가중치 (0~1)")
    parser.add_argument("--save-testset", type=str, default=None, help="생성된 테스트셋 저장 경로")
    parser.add_argument("--objectivity", action="store_true", help="Objectivity Score 평가 추가 실행")
    args = parser.parse_args()

    # ── 1. 테스트셋 준비 ──────────────────────────────────
    if args.testset and Path(args.testset).exists():
        with open(args.testset, encoding="utf-8") as f:
            testset = json.load(f)
        logger.info(f"기존 테스트셋 로드: {len(testset)}개")
    elif args.pdf:
        from rag.pdf import PDFRetrievalChain
        chain = PDFRetrievalChain(source_uri=args.pdf, bm25_weight=args.alpha, dense_weight=1-args.alpha)
        chain.create_chain()

        # 청크에 chunk_id 부여
        for i, doc in enumerate(chain._split_docs):
            doc.metadata["chunk_id"] = i

        testset = generate_testset_with_llm(chain._split_docs, n_samples=args.n_samples)
        logger.info(f"테스트셋 생성 완료: {len(testset)}개")

        if args.save_testset:
            with open(args.save_testset, "w", encoding="utf-8") as f:
                json.dump(testset, f, ensure_ascii=False, indent=2)
            logger.info(f"테스트셋 저장: {args.save_testset}")
    else:
        print("❌ --pdf 또는 --testset 중 하나를 지정하세요.")
        sys.exit(1)

    # ── 2. Retriever 빌드 ─────────────────────────────────
    if args.pdf:
        retriever = chain.retriever
    else:
        # 테스트셋만 있는 경우 PDF 없이 평가 불가
        print("❌ 테스트셋 단독 평가 시 --pdf도 함께 지정하세요.")
        sys.exit(1)

    # ── 3. 평가 실행 ──────────────────────────────────────
    results = evaluate(retriever, testset, k_values=args.k)

    # ── 4. Objectivity Score 평가 (선택) ─────────────────
    if args.objectivity:
        logger.info("Objectivity Score 평가 시작...")
        obj_result = evaluate_objectivity(retriever)
        results["objectivity"] = obj_result

        sep = "=" * 50
        print(f"\n{sep}")
        print("  🎯 Objectivity Score 평가 결과")
        print(sep)
        print(f"  쿼리 수            : {obj_result['n_queries']}개")
        print(f"  반론 문서 포함 비율: {obj_result['risk_coverage_rate']:.1%}  "
              f"{'✅' if obj_result['passed'] else '❌'} (기준: ≥ 50%)")
        if obj_result["pro_only_queries"]:
            print(f"  반론 없는 쿼리({len(obj_result['pro_only_queries'])}개):")
            for q in obj_result["pro_only_queries"][:3]:
                print(f"    - {q}")
        print(sep)

    print_results(results, k_values=args.k)

    # ── 5. 결과 저장 ──────────────────────────────────────
    out_path = Path("outputs") / "eval_results.json"
    Path("outputs").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
