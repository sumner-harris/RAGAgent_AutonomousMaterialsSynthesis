from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from state.config import (
    DEFAULT_RETRIEVAL_DIVERSITY,
    DEFAULT_RETRIEVAL_METHOD,
    DEFAULT_RERANK_TOP_K,
    GRAPHRAG_GLOBAL_RETRIEVAL_METHOD,
    GRAPHRAG_LOCAL_RETRIEVAL_METHOD,
    TOP_K_TEXT_FAISS,
)


class RetrievalMethod(str, Enum):
    """GUI-equivalent retrieval methods exposed through MCP."""

    FAISS_MMR = "faiss_mmr"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_MULTI_QUERY = "hybrid_multi_query"
    HYBRID_MULTI_QUERY_COMPRESSION = "hybrid_multi_query_compression"
    GRAPHRAG_GLOBAL = GRAPHRAG_GLOBAL_RETRIEVAL_METHOD
    GRAPHRAG_LOCAL = GRAPHRAG_LOCAL_RETRIEVAL_METHOD


class KBBuildInput(BaseModel):
    pdf_dir: str
    index_path: str
    meta_path: str
    graphrag_dir: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_profile: str = ""
    chat_model: str = "gpt-4.1-2025-04-14"
    api_key: str = ""
    base_url: str = ""
    embedding_base_url: str = ""
    run_graphrag: bool


class KBBuildResult(BaseModel):
    total_chunks: int
    total_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class KBLoadInput(BaseModel):
    index_path: str
    meta_path: str
    graphrag_dir: str = ""
    api_key: str = ""
    base_url: str = ""
    embedding_base_url: str = ""


class KBLoadResult(BaseModel):
    status: str
    embedding_model: str = "text-embedding-3-small"
    embedding_profile: str = ""
    dimension: int
    chunk_count: int
    graphrag_dir: str = ""


class KBAppendInput(BaseModel):
    index_path: str
    meta_path: str
    append_folder: str
    graphrag_dir: str = ""
    chat_model: str = "gpt-4.1-2025-04-14"
    api_key: str = ""
    base_url: str = ""
    embedding_base_url: str = ""
    run_graphrag: bool


class KBAppendResult(BaseModel):
    new_chunks: int
    new_tokens: int
    estimated_cost: float
    warnings: List[str] = Field(default_factory=list)


class RetrieveContextInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: str = ""
    api_key: str = ""
    base_url: str = ""
    embedding_base_url: str = ""
    diversity: float = Field(
        default=DEFAULT_RETRIEVAL_DIVERSITY,
        ge=0.0,
        le=1.0,
    )
    top_k_faiss: int = TOP_K_TEXT_FAISS
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_top_k: int | None = Field(default=DEFAULT_RERANK_TOP_K, ge=1)
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod(DEFAULT_RETRIEVAL_METHOD),
        description="Retrieval pipeline to run. Matches the methods available in the GUI.",
    )
    use_graphrag: bool = False
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"
    chat_model: str = ""


class SourceRef(BaseModel):
    source: str
    chunk_id: int


class SourceDocumentDetail(BaseModel):
    source: str
    chunk_ids: List[int] = Field(default_factory=list)
    graph_source_ids: List[str] = Field(default_factory=list)
    graph_report_ids: List[str] = Field(default_factory=list)
    graph_entity_ids: List[str] = Field(default_factory=list)
    graph_relationship_ids: List[str] = Field(default_factory=list)
    graph_claim_ids: List[str] = Field(default_factory=list)
    graph_text_unit_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0


class RerankChunk(BaseModel):
    rank: int
    source: str = ""
    chunk_id: int | None = None
    score: float
    text: str = ""


class RetrieveContextResult(BaseModel):
    context_text: str
    retrieved_context_text: str = ""
    rerank_chunks: List[RerankChunk] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    source_document_details: List[SourceDocumentDetail] = Field(
        default_factory=list
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_profile: str = ""


class GenerateAnswerInput(BaseModel):
    query: str
    context_text: str
    model: str
    system: str
    api_key: str = ""
    base_url: str = ""
    enable_web_search: bool


class GenerateAnswerResult(BaseModel):
    answer: str


class RetrieveAndAnswerInput(BaseModel):
    query: str
    index_path: str
    meta_path: str
    graphrag_dir: str = ""
    api_key: str = ""
    base_url: str = ""
    embedding_base_url: str = ""
    diversity: float = Field(
        default=DEFAULT_RETRIEVAL_DIVERSITY,
        ge=0.0,
        le=1.0,
    )
    top_k_faiss: int = TOP_K_TEXT_FAISS
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_top_k: int | None = Field(default=DEFAULT_RERANK_TOP_K, ge=1)
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod(DEFAULT_RETRIEVAL_METHOD),
        description="Retrieval pipeline to run. Matches the methods available in the GUI.",
    )
    use_graphrag: bool = False
    retriever_model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"
    chat_model: str = ""
    model: str = "gpt-4o-mini"
    response_model: str | None = None
    system: str
    enable_web_search: bool


class RetrieveAndAnswerResult(BaseModel):
    answer: str
    context_text: str
    retrieved_context_text: str = ""
    rerank_chunks: List[RerankChunk] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    source_document_details: List[SourceDocumentDetail] = Field(
        default_factory=list
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_profile: str = ""
