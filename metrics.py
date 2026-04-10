"""
성능 지표 트래커 (MetricsTracker)
실행 중 각 노드의 성능을 실시간으로 기록하고 출력합니다.

측정 지표:
- 노드별 실행 시간
- Retrieval Judge 관련성 점수
- TRL Judge 통과/실패 상세
- 반복 횟수 (iteration_count)
- 보고서 완결성 (기술 용어 포함 여부)
- Objectivity Score: 반론/리스크 문서가 검색 결과에 포함된 비율 (확증 편향 방지)
"""

import time
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 보고서 완결성 검증용 필수 기술 용어
REQUIRED_ENTITIES = ["HBM4", "PIM", "CXL", "Samsung", "Micron", "TRL"]


class MetricsTracker:
    """런타임 성능 지표 수집 및 출력"""

    def __init__(self):
        self.start_time = time.time()
        self.records: dict[str, list[dict]] = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record(self, node_name: str, data: dict[str, Any]):
        """노드 실행 지표 기록"""
        if node_name not in self.records:
            self.records[node_name] = []
        entry = {"timestamp": time.time() - self.start_time, **data}
        self.records[node_name].append(entry)
        logger.debug(f"[Metrics] {node_name}: {data}")

    def get_total_elapsed(self) -> float:
        return round(time.time() - self.start_time, 2)

    def compute_entity_recall(self, final_report: str) -> dict:
        """Technical Entity Recall: 필수 기술 용어 포함 여부"""
        found = [e for e in REQUIRED_ENTITIES if re.search(e, final_report, re.IGNORECASE)]
        missing = [e for e in REQUIRED_ENTITIES if e not in found]
        return {
            "recall": len(found) / len(REQUIRED_ENTITIES),
            "found": found,
            "missing": missing,
            "passed": len(missing) == 0,
        }

    def compute_trl_coverage(self) -> dict:
        """TRL 근거 충족률: TRL Judge 통과 이력"""
        trl_records = self.records.get("trl_judge", [])
        if not trl_records:
            return {"coverage": 0.0, "attempts": 0, "passed": False}
        total = len(trl_records)
        passed_count = sum(1 for r in trl_records if r.get("passed", False))
        return {
            "coverage": passed_count / total,
            "attempts": total,
            "passed": trl_records[-1].get("passed", False),
            "detail": trl_records[-1].get("detail", {}),
        }

    def compute_retrieval_quality(self) -> dict:
        """Retrieval Judge 점수 이력"""
        rj_records = self.records.get("retrieval_judge", [])
        if not rj_records:
            return {"avg_score": 0.0, "attempts": 0}
        scores = [r.get("score", 0) for r in rj_records]
        return {
            "avg_score": round(sum(scores) / len(scores), 3),
            "max_score": max(scores),
            "min_score": min(scores),
            "attempts": len(scores),
            "final_passed": rj_records[-1].get("passed", False),
        }

    def compute_objectivity_score(self, rag_docs: list) -> dict:
        """
        Objectivity Score: 반론/리스크 문서가 검색 결과에 포함된 비율
        - risk_ratio ≥ 0.2 (20%) 이상이어야 확증 편향 위험 없음으로 판정
        - source_type 메타데이터가 없는 경우(PDF-less 모드) 측정 불가로 반환
        """
        RISK_TYPES = {"risk", "industry"}
        PRO_TYPES = {"ir_press", "academic", "foundry", "whitepaper", "market"}

        has_metadata = any(
            isinstance(d, dict) and d.get("metadata", {}).get("source_type")
            for d in rag_docs
        )
        if not has_metadata:
            return {
                "risk_doc_count": 0, "pro_doc_count": 0, "total_doc_count": len(rag_docs),
                "risk_ratio": 0.0, "passed": None,
                "note": "source_type 메타데이터 없음 (PDF 없이 실행된 경우)",
            }

        risk_docs = [
            d for d in rag_docs
            if isinstance(d, dict) and d.get("metadata", {}).get("source_type") in RISK_TYPES
        ]
        pro_docs = [
            d for d in rag_docs
            if isinstance(d, dict) and d.get("metadata", {}).get("source_type") in PRO_TYPES
        ]
        total = len(rag_docs)
        risk_ratio = len(risk_docs) / total if total > 0 else 0.0
        return {
            "risk_doc_count": len(risk_docs),
            "pro_doc_count": len(pro_docs),
            "total_doc_count": total,
            "risk_ratio": round(risk_ratio, 3),
            "passed": risk_ratio >= 0.2,
            "note": "risk_ratio < 0.2 시 확증 편향 위험 — 이중 쿼리 또는 반론 PDF 추가 권장",
        }

    def compute_node_timings(self) -> dict:
        """노드별 실행 시간 집계"""
        timings = {}
        for node, records in self.records.items():
            elapsed_list = [r.get("elapsed_sec", 0) for r in records if "elapsed_sec" in r]
            if elapsed_list:
                timings[node] = {
                    "total_sec": round(sum(elapsed_list), 2),
                    "avg_sec": round(sum(elapsed_list) / len(elapsed_list), 2),
                    "calls": len(elapsed_list),
                }
        return timings

    def print_summary(self, final_report: str = ""):
        """전체 성능 지표 콘솔 출력"""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  📊 실행 성능 지표 요약")
        print(sep)

        # 총 실행 시간
        print(f"\n⏱  총 실행 시간: {self.get_total_elapsed()}초")

        # 노드별 타이밍
        timings = self.compute_node_timings()
        if timings:
            print("\n📌 노드별 실행 시간:")
            for node, t in timings.items():
                print(f"   {node:<20}: {t['total_sec']:>6.2f}s (×{t['calls']}회)")

        # Retrieval Judge
        rq = self.compute_retrieval_quality()
        print(f"\n🔍 Retrieval Judge:")
        print(f"   평균 관련성 점수 : {rq.get('avg_score', 0):.3f}")
        print(f"   최종 통과 여부   : {'✅ 통과' if rq.get('final_passed') else '❌ 미달'}")
        print(f"   평가 횟수        : {rq.get('attempts', 0)}회")

        # TRL Judge
        trl = self.compute_trl_coverage()
        print(f"\n🧪 TRL Judge:")
        print(f"   최종 통과 여부   : {'✅ 통과' if trl.get('passed') else '❌ 미달'}")
        print(f"   시도 횟수        : {trl.get('attempts', 0)}회")
        if trl.get("detail"):
            for comp, d in trl["detail"].items():
                status = "✅" if d.get("passed") else "❌"
                print(f"   {comp:<12}: TRL {d.get('trl','?')} | 강력근거 {d.get('strong_evidence',0)}개 | {status}")

        # 반복 횟수
        sup_records = self.records.get("supervisor", [])
        iterations = max([r.get("iteration", 0) for r in sup_records], default=0) if sup_records else 0
        print(f"\n🔄 최대 반복 횟수: {iterations} / 3")

        # Technical Entity Recall
        if final_report:
            er = self.compute_entity_recall(final_report)
            recall_pct = er["recall"] * 100
            status = "✅" if er["passed"] else "⚠️"
            print(f"\n📝 Technical Entity Recall: {recall_pct:.0f}% {status}")
            if er["missing"]:
                print(f"   누락 용어: {', '.join(er['missing'])}")

        # Objectivity Score (확증 편향 방지 지표)
        obj_records = self.records.get("objectivity", [])
        if obj_records:
            obj = obj_records[-1]
            passed = obj.get("passed")
            risk_ratio = obj.get("risk_ratio", 0.0)
            risk_count = obj.get("risk_doc_count", 0)
            total_count = obj.get("total_doc_count", 0)
            if passed is None:
                status_str = "⚪ N/A (메타데이터 없음)"
            elif passed:
                status_str = "✅ 편향 없음"
            else:
                status_str = "⚠️ 확증 편향 위험"
            print(f"\n🎯 Objectivity Score: {status_str}")
            print(f"   반론 문서 비율  : {risk_ratio:.1%} ({risk_count}/{total_count}개)")
            if passed is False:
                print(f"   {obj.get('note', '')}")

        print(f"\n{sep}\n")

    def save_to_file(self, output_dir: str = "outputs", final_report: str = ""):
        """지표를 JSON 파일로 저장"""
        Path(output_dir).mkdir(exist_ok=True)
        metrics_path = Path(output_dir) / f"metrics_{self.run_id}.json"

        # objectivity score는 마지막 기록된 rag_docs 기준으로 재계산
        obj_records = self.records.get("objectivity", [])
        objectivity = obj_records[-1] if obj_records else {}

        summary = {
            "run_id": self.run_id,
            "total_elapsed_sec": self.get_total_elapsed(),
            "node_timings": self.compute_node_timings(),
            "retrieval_quality": self.compute_retrieval_quality(),
            "trl_coverage": self.compute_trl_coverage(),
            "entity_recall": self.compute_entity_recall(final_report) if final_report else {},
            "objectivity_score": objectivity,
            "raw_records": self.records,
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[Metrics] 지표 저장 완료: {metrics_path}")
        return str(metrics_path)
