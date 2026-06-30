import os

import streamlit as st

from state.config import (
    BM25_RETRIEVAL_METHODS,
    GRAPHRAG_RETRIEVAL_METHODS,
    TOP_K_TEXT_FAISS,
    TOP_K_UPLOAD_FAISS,
)
from state.schemas import RetrievalRequest
from RAG.retrieval.graphrag_query import (
    GraphRAGQueryError,
    graphrag_query_method_for_retrieval_method,
    query_graphrag_with_context,
)
from RAG.retrieval.pipeline import (
    build_bm25_retriever_from_metadata,
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


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _append_document_context_lines(context_lines: list[str], documents) -> None:
    """Append formatted document lines to a context list."""
    context_lines.extend(
        f"[{document.metadata['source']} | chunk {document.metadata['chunk_id']}]: "
        f"{document.page_content}"
        for document in documents
    )


class QAContextRetriever:
    """Retrieves context using exactly one selected retrieval method."""

    def _maybe_rerank(self, *, query: str, documents, callbacks=None):
        """Apply reranking when a rerank endpoint is configured."""
        if not reranking_is_configured(
            model=st.session_state.get("rerank_model"),
            base_url=st.session_state.get("rerank_api_base_url"),
        ):
            return documents

        try:
            reranked_documents = rerank_documents(
                query=query,
                documents=documents,
                model=st.session_state.get("rerank_model", ""),
                api_key=st.session_state.get("api_key"),
                base_url=st.session_state.get("rerank_api_base_url"),
            )
            return keep_top_reranked_documents(
                documents=reranked_documents,
                top_k=st.session_state.get("rerank_top_k"),
            )
        except Exception as exc:
            _call(
                _get_cb(callbacks, "warning"),
                f"Rerank failed; using base retrieval order instead: {exc}",
            )
            return documents

    def retrieve(
        self,
        query: str,
        use_uploads: bool = True,
        retrieval_method: str | None = None,
        callbacks=None,
    ) -> dict:
        """Retrieve answer context plus optional rerank details.

        Args:
            query: User question text.
            use_uploads: Whether uploaded files should be searched too.
            retrieval_method: Explicit retrieval method to run.
            callbacks: Optional UI callback sink for info/warning messages.

        Returns:
            Dictionary containing `context_text` for generation,
            `retrieved_context_text` for the UI display, rerank details,
            and source-document provenance.
        """
        method = resolve_retrieval_method(retrieval_method)
        req = RetrievalRequest(
            query=query,
            use_uploads=use_uploads,
            retrieval_method=method,
            top_k_faiss=TOP_K_TEXT_FAISS,
            top_k_uploads=TOP_K_UPLOAD_FAISS,
            diversity=st.session_state.diversity,
        )

        context_lines = []
        retrieved_context_lines = []
        rerank_chunks = []
        source_documents = []
        source_document_details = []

        if req.retrieval_method not in GRAPHRAG_RETRIEVAL_METHODS:
            docs = retrieve_documents_with_method(
                query,
                req.retrieval_method,
                vector_store=st.session_state.db,
                bm25_retriever=st.session_state.bm25,
                top_k=req.top_k_faiss,
                diversity=req.diversity,
                api_key=st.session_state.api_key,
                base_url=st.session_state.api_base_url,
                retriever_model=st.session_state.gpt_model or "gpt-4o-mini",
                compressor_model=st.session_state.gpt_model or "gpt-4o-mini",
            )

            upload_docs = []
            upload_meta = st.session_state.get("upload_meta", [])
            if req.use_uploads and upload_meta:
                upload_bm25 = None
                if req.retrieval_method in BM25_RETRIEVAL_METHODS:
                    upload_bm25 = build_bm25_retriever_from_metadata(
                        upload_meta,
                        k=req.top_k_uploads,
                    )
                upload_docs = retrieve_documents_with_method(
                    query,
                    req.retrieval_method,
                    vector_store=st.session_state.get("upload_db"),
                    bm25_retriever=upload_bm25,
                    top_k=req.top_k_uploads,
                    diversity=req.diversity,
                    api_key=st.session_state.api_key,
                    base_url=st.session_state.api_base_url,
                    retriever_model=st.session_state.gpt_model or "gpt-4o-mini",
                    compressor_model=st.session_state.gpt_model or "gpt-4o-mini",
                )

            merged_docs = list(docs)
            if upload_docs:
                merged_docs.extend(upload_docs)
            _append_document_context_lines(retrieved_context_lines, merged_docs)
            merged_docs = self._maybe_rerank(
                query=query,
                documents=merged_docs,
                callbacks=callbacks,
            )
            rerank_chunks = summarize_reranked_documents(documents=merged_docs)

            _append_document_context_lines(context_lines, merged_docs)
            source_documents, source_document_details = summarize_langchain_documents(
                merged_docs
            )
        elif req.use_uploads and st.session_state.get("upload_meta"):
            _call(
                _get_cb(callbacks, "info"),
                "GraphRAG mode does not retrieve uploaded text chunks. Uploaded images are still included.",
            )

        if req.use_uploads and st.session_state.get("upload_images"):
            for img in st.session_state.upload_images:
                image_line = f"[uploaded/{img['name']} | image]: (image attached)"
                context_lines.append(image_line)
                retrieved_context_lines.append(image_line)

        if req.retrieval_method in GRAPHRAG_RETRIEVAL_METHODS:
            graph_root = st.session_state.get("graphrag")
            if graph_root and os.path.isdir(graph_root):
                try:
                    graph_result = query_graphrag_with_context(
                        graph_root,
                        query,
                        method=graphrag_query_method_for_retrieval_method(
                            req.retrieval_method
                        ),
                        api_key=st.session_state.get("api_key"),
                        base_url=st.session_state.get("api_base_url"),
                        chat_model=st.session_state.get("gpt_model"),
                        embedding_base_url=st.session_state.get(
                            "embedding_api_base_url"
                        ),
                        embedding_model=st.session_state.get("embedding_model"),
                    )
                    source_documents = graph_result.source_documents
                    source_document_details = graph_result.source_document_details
                    if graph_result.answer_text:
                        graph_line = "\n[GraphRAG Context]:\n" + graph_result.answer_text
                        context_lines.append(graph_line)
                        retrieved_context_lines.append(graph_line)
                    elif graph_result.source_documents:
                        _call(
                            _get_cb(callbacks, "info"),
                            "GraphRAG returned source provenance, but no answer text.",
                        )
                    else:
                        _call(
                            _get_cb(callbacks, "info"),
                            "No Knowledge-Graph context found for this query.",
                        )
                except GraphRAGQueryError as exc:
                    _call(
                        _get_cb(callbacks, "warning"),
                        f"Error while retrieving from Knowledge-Graph: {exc}",
                    )
            else:
                _call(
                    _get_cb(callbacks, "info"),
                    "No Knowledge-Graph directory loaded - skipping graph-based retrieval.",
                )

        return {
            "context_text": "\n\n".join(context_lines),
            "retrieved_context_text": "\n\n".join(retrieved_context_lines),
            "rerank_chunks": rerank_chunks,
            "source_documents": source_documents,
            "source_document_details": source_document_details,
        }
