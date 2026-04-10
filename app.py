"""
app.py — HBM4/PIM/CXL 기술 전략 분석 AI Agent 진입점
설계서 B. Architecture (Supervisor 패턴) 기반 StateGraph 조립

실행:
    python app.py --query "HBM4 Samsung Micron 2025 경쟁 동향 분석"
    python app.py  # 기본 쿼리 사용
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/run.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState, initial_state
from metrics import MetricsTracker

from agents.supervisor import supervisor_node, MAX_ITERATIONS
from agents.rag_agent import rag_agent_node
from agents.web_agent import web_agent_node
from agents.judges import retrieval_judge_node, trl_judge_node
from agents.analysis_agent import analysis_agent_node
from agents.draft_agent import draft_agent_node
from agents.formatting_node import formatting_node


def build_graph(metrics: MetricsTracker) -> StateGraph:
    """StateGraph 조립: 설계서 flowchart 구현"""

    # ── 노드 래퍼 (metrics 주입) ───────────────────────────
    def _supervisor(state): return supervisor_node(state, metrics)
    def _rag(state): return rag_agent_node(state, metrics)
    def _web(state): return web_agent_node(state, metrics)
    def _retrieval_judge(state): return retrieval_judge_node(state, metrics)
    def _analysis(state): return analysis_agent_node(state, metrics)
    def _trl_judge(state): return trl_judge_node(state, metrics)
    def _draft(state): return draft_agent_node(state, metrics)
    def _format(state): return formatting_node(state, metrics)

    # ── GraphBuilder ───────────────────────────────────────
    builder = StateGraph(ResearchState)

    # 노드 등록
    builder.add_node("supervisor", _supervisor)
    builder.add_node("rag_search", _rag)
    builder.add_node("web_search", _web)
    builder.add_node("retrieval_judge", _retrieval_judge)
    builder.add_node("analysis", _analysis)
    builder.add_node("trl_judge", _trl_judge)
    builder.add_node("draft", _draft)
    builder.add_node("format", _format)

    # 진입점: 항상 Supervisor부터
    builder.set_entry_point("supervisor")

    # Supervisor → conditional edge (next_action 기반 라우팅)
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_action", "end"),
        {
            "rag_search":       "rag_search",
            "web_search":       "web_search",
            "retrieval_judge":  "retrieval_judge",
            "analysis":         "analysis",
            "trl_judge":        "trl_judge",
            "draft":            "draft",
            "format":           "format",
            "end":              END,
        },
    )

    # 각 하위 에이전트 → Supervisor (중앙집중 제어, 설계서 원칙)
    for node in ["rag_search", "web_search", "retrieval_judge",
                 "analysis", "trl_judge", "draft"]:
        builder.add_edge(node, "supervisor")

    # format → END (Supervisor 재확인 패턴)
    builder.add_edge("format", "supervisor")

    return builder.compile(checkpointer=MemorySaver())


def run(query: str, verbose: bool = True) -> dict:
    """파이프라인 실행 메인 함수"""
    os.makedirs("outputs", exist_ok=True)

    metrics = MetricsTracker()
    graph = build_graph(metrics)

    state = initial_state(query)
    config = {"configurable": {"thread_id": metrics.run_id}}

    logger.info(f"[App] 실행 시작: {query[:80]}")
    logger.info(f"[App] Run ID: {metrics.run_id}")

    try:
        final_state = graph.invoke(state, config=config)
    except Exception as e:
        logger.error(f"[App] 그래프 실행 오류: {e}")
        raise

    final_report = final_state.get("final_report", "")

    # ── 성능 지표 출력 ────────────────────────────────────
    if verbose:
        metrics.print_summary(final_report)

    # ── 지표 파일 저장 ────────────────────────────────────
    metrics_path = metrics.save_to_file("outputs", final_report)
    logger.info(f"[App] 지표 저장: {metrics_path}")

    # ── 결과 요약 출력 ────────────────────────────────────
    if verbose and final_report:
        print("\n" + "=" * 60)
        print("  📄 최종 보고서 (앞 500자 미리보기)")
        print("=" * 60)
        print(final_report[:500])
        print("...\n[전체 보고서는 outputs/ 디렉토리 확인]")

    return {
        "final_report": final_report,
        "iteration_count": final_state.get("iteration_count", 0),
        "metrics_path": metrics_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="HBM4/PIM/CXL 기술 전략 분석 AI Agent"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="HBM4, PIM, CXL 분야에서 Samsung과 Micron의 2024-2025년 R&D 동향과 TRL 현황을 분석해주세요.",
        help="분석 질의 (기본값: HBM4/PIM/CXL 동향 분석)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="성능 지표 출력 생략"
    )
    args = parser.parse_args()

    result = run(query=args.query, verbose=not args.quiet)
    sys.exit(0 if result["final_report"] else 1)


if __name__ == "__main__":
    main()
