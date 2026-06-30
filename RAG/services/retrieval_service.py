import pickle
from pathlib import Path

import faiss
from langchain_community.retrievers import BM25Retriever

from ingestion.embedding_profiles import (
    create_embedding_function,
    resolve_embedding_profile_name,
)
from RAG.openai_compat import connection_is_configured
from RAG.retrieval.graphrag_query import (
    GraphRAGQueryError,
    graphrag_query_method_for_retrieval_method,
    query_graphrag_with_context,
)
from RAG.retrieval.pipeline import (
    build_documents_from_metadata,
    resolve_retrieval_method,
    retrieve_documents_with_method,
)
from RAG.retrieval.provenance import summarize_langchain_documents
from RAG.retrieval.reranker import (
    keep_top_reranked_documents,
    rerank_documents,
    reranking_is_configured,
    summarize_reranked_documents,
)
from state.config import (
    BM25_RETRIEVAL_METHODS,
    COMPRESSED_RETRIEVAL_METHODS,
    DEFAULT_RETRIEVAL_DIVERSITY,
    DEFAULT_RERANK_TOP_K,
    GRAPHRAG_RETRIEVAL_METHODS,
    MULTI_QUERY_RETRIEVAL_METHODS,
    TOP_K_TEXT_BM25,
    TOP_K_TEXT_FAISS,
    VECTOR_RETRIEVAL_METHODS,
)


def _load_index_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    if not metadata:
        raise ValueError("Metadata file is empty.")
    return index, metadata


def _build_faiss_store(index, metadata, embeddings, docs):
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    ids = [str(i) for i in range(len(metadata))]
    docs_dict = {ids[i]: docs[i] for i in range(len(metadata))}
    docstore = InMemoryDocstore(docs_dict)
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def _append_document_context_lines(context_lines: list[str], documents) -> None:
    """Append formatted document lines to a context list."""
    context_lines.extend(
        f"[{document.metadata['source']} | chunk {document.metadata['chunk_id']}]: "
        f"{document.page_content}"
        for document in documents
    )


def _build_chunk_source_refs(documents) -> list[dict]:
    """Return chunk-level source references for retrieved LangChain documents."""
    return [
        {
            "source": document.metadata["source"],
            "chunk_id": document.metadata["chunk_id"],
        }
        for document in documents
    ]


def _method_requires_prompt_connection(method: str) -> bool:
    """Return whether a retrieval method needs an LLM prompt route."""
    return (
        method in MULTI_QUERY_RETRIEVAL_METHODS
        or method in COMPRESSED_RETRIEVAL_METHODS
    )


def _method_requires_embedding_connection(method: str) -> bool:
    """Return whether a retrieval method needs an embedding route."""
    return method in VECTOR_RETRIEVAL_METHODS


def _validate_retrieval_connections(
    *,
    method: str,
    api_key: str | None,
    base_url: str | None,
    embedding_base_url: str | None,
) -> None:
    """Validate only the endpoint types required by the selected retrieval method."""
    if method in GRAPHRAG_RETRIEVAL_METHODS:
        return

    errors = []
    if _method_requires_prompt_connection(method) and not connection_is_configured(
        api_key,
        base_url,
    ):
        errors.append(
            "A prompt model endpoint must be configured for this retrieval method."
        )
    if _method_requires_embedding_connection(
        method
    ) and not connection_is_configured(
        api_key,
        embedding_base_url,
    ):
        errors.append(
            "An embedding endpoint must be configured for this retrieval method."
        )

    if errors:
        raise ValueError(" ".join(errors))


def retrieve_context(
    *,
    query: str,
    index_path: str,
    meta_path: str,
    graphrag_dir: str | None,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_base_url: str | None = None,
    rerank_base_url: str | None = None,
    rerank_model: str | None = None,
    rerank_top_k: int | None = DEFAULT_RERANK_TOP_K,
    diversity: float = DEFAULT_RETRIEVAL_DIVERSITY,
    top_k_faiss: int | None = None,
    retrieval_method: str | None = None,
    use_graphrag: bool = False,
    retriever_model: str = "gpt-4o-mini",
    compressor_model: str = "gpt-4o-mini",
    chat_model: str | None = None,
):
    """Retrieve context from a saved knowledge base using shared defaults.

    Args:
        query: User question text.
        index_path: Path to the saved FAISS index.
        meta_path: Path to the saved chunk metadata pickle.
        graphrag_dir: Optional GraphRAG workspace directory.
        api_key: Optional API key for prompt and rerank endpoints.
        base_url: Optional prompt-model base URL.
        embedding_base_url: Optional embedding-model base URL.
        rerank_base_url: Optional rerank endpoint base URL.
        rerank_model: Optional rerank model identifier.
        rerank_top_k: Maximum reranked chunks to keep. Defaults to shared config.
        diversity: FAISS/MMR diversity setting. Defaults to shared config.
        top_k_faiss: Candidate vector-retrieval count before reranking.
        retrieval_method: Selected retrieval method name.
        use_graphrag: Whether to force GraphRAG when no retrieval method is supplied.
        retriever_model: LLM used for multi-query expansion.
        compressor_model: LLM used for contextual compression.
        chat_model: Prompt model ID to use for GraphRAG query-time generation.

    Returns:
        Dictionary with `context_text` for generation,
        `retrieved_context_text` for UI display, rerank details, and source metadata.
    """
    method = resolve_retrieval_method(
        retrieval_method,
        use_graphrag=use_graphrag,
    )
    _validate_retrieval_connections(
        method=method,
        api_key=api_key,
        base_url=base_url,
        embedding_base_url=embedding_base_url,
    )

    index, metadata = _load_index_and_metadata(index_path, meta_path)
    embedding_model = metadata[0]["embedding_model"]
    embedding_profile = metadata[0].get("embedding_profile") or (
        resolve_embedding_profile_name(model_name=embedding_model)
    )

    docs = build_documents_from_metadata(metadata)
    context_lines = []
    retrieved_context_lines = []
    rerank_chunks = []
    sources = []
    source_documents = []
    source_document_details = []
    top_k = top_k_faiss or TOP_K_TEXT_FAISS

    if method not in GRAPHRAG_RETRIEVAL_METHODS:
        vector_store = None
        if method in VECTOR_RETRIEVAL_METHODS:
            embeddings = create_embedding_function(
                model=embedding_model,
                profile_name=embedding_profile,
                api_key=api_key,
                base_url=embedding_base_url,
                fallback_to_env=False,
            )
            vector_store = _build_faiss_store(index, metadata, embeddings, docs)
        bm25 = None
        if method in BM25_RETRIEVAL_METHODS:
            bm25 = BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25)

        retrieved_docs = retrieve_documents_with_method(
            query,
            method,
            vector_store=vector_store,
            bm25_retriever=bm25,
            top_k=top_k,
            diversity=diversity,
            api_key=api_key,
            base_url=base_url,
            retriever_model=retriever_model,
            compressor_model=compressor_model,
        )
        _append_document_context_lines(retrieved_context_lines, retrieved_docs)
        if reranking_is_configured(model=rerank_model, base_url=rerank_base_url):
            try:
                retrieved_docs = rerank_documents(
                    query=query,
                    documents=list(retrieved_docs),
                    model=rerank_model or "",
                    api_key=api_key,
                    base_url=rerank_base_url,
                )
                retrieved_docs = keep_top_reranked_documents(
                    documents=retrieved_docs,
                    top_k=rerank_top_k,
                )
            except Exception:
                pass
        rerank_chunks = summarize_reranked_documents(documents=retrieved_docs)

        _append_document_context_lines(context_lines, retrieved_docs)
        sources.extend(_build_chunk_source_refs(retrieved_docs))
        source_documents, source_document_details = summarize_langchain_documents(
            retrieved_docs
        )

    if method in GRAPHRAG_RETRIEVAL_METHODS and graphrag_dir:
        root = Path(graphrag_dir)
        if root.is_dir():
            try:
                graph_result = query_graphrag_with_context(
                    root,
                    query,
                    method=graphrag_query_method_for_retrieval_method(method),
                    api_key=api_key,
                    base_url=base_url,
                    chat_model=chat_model,
                    embedding_base_url=embedding_base_url,
                    embedding_model=embedding_model,
                )
                source_documents = graph_result.source_documents
                source_document_details = graph_result.source_document_details
                if graph_result.answer_text:
                    graph_line = "\n[GraphRAG Context]:\n" + graph_result.answer_text
                    context_lines.append(graph_line)
                    retrieved_context_lines.append(graph_line)
                    sources.append({"source": "graphrag", "chunk_id": -1})
                elif graph_result.source_documents:
                    sources.append({"source": "graphrag", "chunk_id": -1})
            except GraphRAGQueryError:
                pass

    return {
        "context_text": "\n\n".join(context_lines),
        "retrieved_context_text": "\n\n".join(retrieved_context_lines),
        "rerank_chunks": rerank_chunks,
        "sources": sources,
        "source_documents": source_documents,
        "source_document_details": source_document_details,
        "embedding_model": embedding_model,
        "embedding_profile": embedding_profile,
    }
