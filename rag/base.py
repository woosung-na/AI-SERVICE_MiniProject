"""
RAG Base Module
- FAISS + BM25 Ensemble Retriever
- CacheBackedEmbeddings (text-embedding-3-small)
- 기존 langgraph-v1/11-RAG/rag/base.py 기반 확장
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)


class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 8
        self.model_name = "gpt-4.1-mini"
        self.temperature = 0
        self.embeddings = "text-embedding-3-small"
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")
        # BM25 vs Dense 가중치 (α=0.5 기본값, eval/로 최적화)
        self.bm25_weight = 0.5
        self.dense_weight = 0.5

    @abstractmethod
    def load_documents(self, source_uris):
        pass

    @abstractmethod
    def create_text_splitter(self):
        pass

    def split_documents(self, docs, text_splitter):
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            underlying_embeddings = OpenAIEmbeddings(model=self.embeddings)
            store = LocalFileStore(str(self.cache_dir))
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings,
                store,
                namespace=self.embeddings,
                key_encoder="sha256",
            )
            return cached_embeddings
        except Exception as e:
            logger.warning(f"CacheBackedEmbeddings 실패, 기본 임베딩 사용: {e}")
            return OpenAIEmbeddings(model=self.embeddings)

    def create_vectorstore(self, split_docs):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        doc_contents = "\n".join([doc.page_content for doc in split_docs])
        doc_hash = hashlib.md5(doc_contents.encode()).hexdigest()
        hash_file = self.index_dir / "doc_hash.txt"
        index_path = str(self.index_dir / "faiss_index")

        try:
            if (
                hash_file.exists()
                and Path(index_path + ".faiss").exists()
                and hash_file.read_text().strip() == doc_hash
            ):
                vectorstore = FAISS.load_local(
                    index_path,
                    self.create_embedding(),
                    allow_dangerous_deserialization=True,
                )
                logger.info("기존 FAISS 인덱스 캐시 로드 완료")
                return vectorstore
        except Exception as e:
            logger.warning(f"캐시 인덱스 로드 실패, 새로 생성: {e}")

        vectorstore = FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )
        try:
            vectorstore.save_local(index_path)
            hash_file.write_text(doc_hash)
            logger.info("FAISS 인덱스 캐시 저장 완료")
        except Exception as e:
            logger.warning(f"인덱스 저장 실패: {e}")
        return vectorstore

    def create_ensemble_retriever(self, split_docs, vectorstore):
        """BM25 + Dense Ensemble Retriever 생성 (설계서 권장 방식)"""
        # Sparse: BM25
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = self.k

        # Dense: FAISS
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[self.bm25_weight, self.dense_weight],
        )
        logger.info(
            f"Ensemble Retriever 생성 완료 "
            f"(BM25={self.bm25_weight}, Dense={self.dense_weight})"
        )
        return ensemble_retriever

    def create_chain(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_ensemble_retriever(split_docs, self.vectorstore)
        self._split_docs = split_docs
        return self
