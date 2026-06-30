from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from RAG.openai_compat import build_openai_client_kwargs
from state.config import (
    BM25_RETRIEVAL_METHODS,
    COMPRESSED_RETRIEVAL_METHODS,
    DEFAULT_RETRIEVAL_DIVERSITY,
    DEFAULT_RETRIEVAL_METHOD,
    GRAPHRAG_GLOBAL_RETRIEVAL_METHOD,
    GRAPHRAG_RETRIEVAL_METHODS,
    LEGACY_RETRIEVAL_METHOD_ALIASES,
    MULTI_QUERY_RETRIEVAL_METHODS,
    RETRIEVAL_METHODS,
    VECTOR_RETRIEVAL_METHODS,
)


def resolve_retrieval_method(retrieval_method=None, *, use_graphrag=False):
    """Return a supported retrieval method with legacy aliases normalized."""
    method = retrieval_method or DEFAULT_RETRIEVAL_METHOD
    if use_graphrag and not retrieval_method:
        method = GRAPHRAG_GLOBAL_RETRIEVAL_METHOD
    method = LEGACY_RETRIEVAL_METHOD_ALIASES.get(method, method)
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unknown retrieval method: {method}")
    return method


def build_documents_from_metadata(text_meta):
    return [
        Document(
            page_content=meta["text"],
            metadata={
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "original_content": meta.get("original_content", meta["text"]),
            },
        )
        for meta in text_meta
    ]


def build_bm25_retriever_from_metadata(text_meta, k):
    docs = build_documents_from_metadata(text_meta)
    if not docs:
        return None
    return BM25Retriever.from_documents(docs, k=k)


def retrieve_documents_with_method(
    query,
    retrieval_method,
    *,
    vector_store=None,
    bm25_retriever=None,
    top_k=50,
    diversity=DEFAULT_RETRIEVAL_DIVERSITY,
    api_key=None,
    base_url=None,
    retriever_model="gpt-4o-mini",
    compressor_model="gpt-4o-mini",
):
    method = resolve_retrieval_method(retrieval_method)
    if method in GRAPHRAG_RETRIEVAL_METHODS:
        return []

    retriever = None
    vector_retriever = None

    if method in VECTOR_RETRIEVAL_METHODS:
        if vector_store is None:
            return []
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 2,
                "lambda_mult": 1.0 - diversity,
            },
        )
        retriever = vector_retriever

    if method in BM25_RETRIEVAL_METHODS:
        if bm25_retriever is None:
            return []
        if method == "bm25":
            retriever = bm25_retriever
        else:
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )

    if retriever is None:
        return []

    if method in MULTI_QUERY_RETRIEVAL_METHODS:
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=ChatOpenAI(
                model=retriever_model,
                **build_openai_client_kwargs(api_key=api_key, base_url=base_url),
            ),
            include_original=True,
        )

    if method in COMPRESSED_RETRIEVAL_METHODS:
        compressor = LLMChainExtractor.from_llm(
            ChatOpenAI(
                model=compressor_model,
                temperature=0,
                **build_openai_client_kwargs(api_key=api_key, base_url=base_url),
            )
        )
        retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=compressor,
        )

    results = retriever.invoke(query)
    if not results:
        return []

    return LongContextReorder().transform_documents(results)
