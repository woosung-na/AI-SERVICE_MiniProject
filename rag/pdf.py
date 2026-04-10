"""
PDF RAG Chain
- PDFPlumber 기반 로더
- RecursiveCharacterTextSplitter
- 파일명 기반 source_type 메타데이터 자동 태깅 (확증 편향 방지)
- 기존 langgraph-v1/11-RAG/rag/pdf.py 기반 확장
"""

from rag.base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Annotated, Tuple
from pathlib import Path
import os
import re
import hashlib
import logging

logger = logging.getLogger(__name__)

# ── 파일명 → source_type 매핑 규칙 ────────────────────────────
# (pattern, source_type, source_category_kr)
_SOURCE_RULES: List[Tuple[str, str, str]] = [
    # 1. 반론/리스크 문서 (최우선 — 이름에 Samsung이 있어도 Risk면 risk)
    (r"Risk|JEDEC|Barriers|IEA|DigitalToday|Energy_Demand", "risk", "반론_리스크"),
    # 2. 학술 논문 / 기술 백서
    (r"ArXiv|Whitepaper|NFI|Metrology|Academic|Paper_20", "academic", "학술_백서"),
    # 3. 금융/시장 분석 (AiMX_TCO·Eugene은 SKHynix_ 파일에 포함 → 제조사보다 먼저)
    #    단, Earnings는 Micron_ 파일에 포함되므로 여기서는 제외하고 제조사 이후에 처리
    (r"TrendForce|Market|TCO|Eugene|Financial", "market", "금융_시장"),
    # 4. 제조사 IR / 보도자료 (Samsung_, SKHynix_, Micron_ 포함)
    (r"Samsung_|SKHynix_|Micron_", "ir_press", "제조사_IR"),
    # 5. 파운드리
    (r"TSMC|CoWoS|BaseDie|Foundry", "foundry", "파운드리"),
    # 6. 산업 트렌드
    (r"Industry_|Adoption_Report|CXL.*Report|CXL.*Adoption", "industry", "산업_트렌드"),
    # 7. 나머지 시장 보고서 (Earnings 등 — 제조사 IR 이후)
    (r"Earnings|Market_Report", "market", "금융_시장"),
]
_DEFAULT_SOURCE_TYPE = ("general", "기타")


def _detect_source_type(filename: str) -> Tuple[str, str]:
    """파일명 패턴으로 (source_type, source_category) 반환"""
    name = Path(filename).name
    for pattern, stype, scat in _SOURCE_RULES:
        if re.search(pattern, name, re.IGNORECASE):
            return stype, scat
    return _DEFAULT_SOURCE_TYPE


class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[List[str], "PDF 파일 경로 리스트"], **kwargs):
        super().__init__()
        self.source_uri = source_uri

        # 다중 PDF → 통합 해시 기반 캐시 경로
        if isinstance(source_uri, list) and len(source_uri) == 1:
            file_hash = hashlib.md5(source_uri[0].encode()).hexdigest()[:8]
            file_name = Path(source_uri[0]).stem
            cache_suffix = f"{file_name}_{file_hash}"
        else:
            combined = "_".join(sorted(source_uri)) if isinstance(source_uri, list) else str(source_uri)
            cache_suffix = f"multi_{hashlib.md5(combined.encode()).hexdigest()[:8]}"

        self.cache_dir = Path(f".cache/embeddings/{cache_suffix}")
        self.index_dir = Path(f".cache/faiss_index/{cache_suffix}")
        logger.info(f"PDF Cache: embeddings={self.cache_dir}, faiss={self.index_dir}")

        # kwargs override
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load_documents(self, source_uris: List[str]) -> List[Document]:
        docs = []
        failed = []
        for uri in source_uris:
            path = Path(uri)
            if not path.exists():
                logger.warning(f"파일 없음: {uri}")
                failed.append(uri)
                continue
            if not uri.lower().endswith(".pdf"):
                logger.warning(f"PDF 아님: {uri}")
                failed.append(uri)
                continue
            try:
                loader = PDFPlumberLoader(uri)
                loaded = loader.load()
                # ── source_type 메타데이터 주입 (확증 편향 방지 핵심) ──
                stype, scat = _detect_source_type(uri)
                for doc in loaded:
                    doc.metadata["source_type"] = stype
                    doc.metadata["source_category"] = scat
                    doc.metadata["source_file"] = Path(uri).name
                docs.extend(loaded)
                logger.info(f"로드 완료: {uri} ({len(loaded)} 페이지) [source_type={stype}]")
            except Exception as e:
                logger.error(f"로드 실패 {uri}: {e}")
                failed.append(uri)

        logger.info(f"로딩 결과: 성공={len(source_uris)-len(failed)}, 실패={len(failed)}, 총={len(docs)}페이지")
        if not docs:
            raise ValueError("로드된 문서 없음. data/ 디렉토리에 PDF를 추가해주세요.")
        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
