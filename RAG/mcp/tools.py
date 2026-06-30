from langchain_core.embeddings import Embeddings

from ingestion.embedding_profiles import create_embedding_function
from RAG.mcp.schemas import (
    GenerateAnswerInput,
    GenerateAnswerResult,
    KBAppendInput,
    KBAppendResult,
    KBBuildInput,
    KBBuildResult,
    KBLoadInput,
    KBLoadResult,
    RetrieveAndAnswerInput,
    RetrieveAndAnswerResult,
    RetrieveContextInput,
    RetrieveContextResult,
)
from RAG.openai_compat import (
    connection_is_configured,
    make_openai_client,
)
from RAG.services.generation_service import generate_answer
from RAG.services.kb_service import append_kb, build_kb, load_kb
from RAG.services.retrieval_service import retrieve_context
from state.config import ENC


def _require_prompt_connection(api_key: str | None, base_url: str | None):
    if not connection_is_configured(api_key, base_url):
        raise ValueError("A prompt model endpoint must be configured.")


def _require_embedding_connection(api_key: str | None, base_url: str | None):
    if not connection_is_configured(api_key, base_url):
        raise ValueError("An embedding endpoint must be configured.")


def _require_graphrag_connection_if_enabled(
    *,
    run_graphrag: bool,
    graphrag_dir: str,
    api_key: str | None,
    base_url: str | None,
):
    if run_graphrag and graphrag_dir:
        _require_prompt_connection(api_key, base_url)


def _make_optional_client(
    api_key: str | None,
    base_url: str | None,
):
    if not connection_is_configured(api_key, base_url):
        return None
    return make_openai_client(api_key=api_key, base_url=base_url)


def _make_embeddings(
    embedding_model: str,
    embedding_profile: str,
    api_key: str,
    base_url: str,
) -> Embeddings:
    return create_embedding_function(
        model=embedding_model,
        profile_name=embedding_profile or None,
        api_key=api_key,
        base_url=base_url,
        fallback_to_env=False,
    )


def _resolve_response_model(params: RetrieveAndAnswerInput) -> str:
    """Return the explicit response model, honoring the legacy alias."""
    model = (params.response_model or params.model).strip()
    if not model:
        raise ValueError("A response model must be configured.")
    return model


def register_tools(mcp):
    def kb_build(params: KBBuildInput) -> KBBuildResult:
        """Build a FAISS knowledge base from PDFs and optionally run GraphRAG."""
        _require_embedding_connection(params.api_key, params.embedding_base_url)
        _require_graphrag_connection_if_enabled(
            run_graphrag=params.run_graphrag,
            graphrag_dir=params.graphrag_dir,
            api_key=params.api_key,
            base_url=params.base_url,
        )
        embeddings = _make_embeddings(
            params.embedding_model,
            params.embedding_profile,
            params.api_key,
            params.embedding_base_url,
        )
        result = build_kb(
            client=_make_optional_client(params.api_key, params.base_url),
            embeddings=embeddings,
            enc=ENC,
            pdf_dir=params.pdf_dir,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            embedding_model=params.embedding_model,
            embedding_profile=params.embedding_profile or None,
            run_graphrag=params.run_graphrag,
            api_key=params.api_key,
            base_url=params.base_url,
            embedding_base_url=params.embedding_base_url,
            chat_model=params.chat_model,
        )
        return KBBuildResult(**result)

    def kb_load(params: KBLoadInput) -> KBLoadResult:
        """Validate and inspect an existing knowledge base on disk."""
        result = load_kb(
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
        )
        return KBLoadResult(**result)

    def kb_append(params: KBAppendInput) -> KBAppendResult:
        """Append new PDFs to an existing knowledge base on disk."""
        _require_embedding_connection(params.api_key, params.embedding_base_url)
        _require_graphrag_connection_if_enabled(
            run_graphrag=params.run_graphrag,
            graphrag_dir=params.graphrag_dir,
            api_key=params.api_key,
            base_url=params.base_url,
        )
        result = append_kb(
            client=_make_optional_client(params.api_key, params.base_url),
            enc=ENC,
            index_path=params.index_path,
            meta_path=params.meta_path,
            append_folder=params.append_folder,
            graphrag_dir=params.graphrag_dir,
            run_graphrag=params.run_graphrag,
            api_key=params.api_key,
            base_url=params.base_url,
            embedding_base_url=params.embedding_base_url,
            chat_model=params.chat_model,
        )
        return KBAppendResult(**result)

    def retrieve_context_tool(params: RetrieveContextInput) -> RetrieveContextResult:
        """Retrieve context and source references from a knowledge base."""
        result = retrieve_context(
            query=params.query,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            api_key=params.api_key,
            base_url=params.base_url,
            embedding_base_url=params.embedding_base_url,
            rerank_base_url=params.rerank_base_url,
            rerank_model=params.rerank_model,
            rerank_top_k=params.rerank_top_k,
            diversity=params.diversity,
            top_k_faiss=params.top_k_faiss,
            retrieval_method=params.retrieval_method,
            use_graphrag=params.use_graphrag,
            retriever_model=params.retriever_model,
            compressor_model=params.compressor_model,
            chat_model=params.chat_model,
        )
        return RetrieveContextResult(**result)

    def generate_answer_tool(params: GenerateAnswerInput) -> GenerateAnswerResult:
        """Generate an answer given a query and context text."""
        answer = generate_answer(
            query=params.query,
            context_text=params.context_text,
            model=params.model,
            system=params.system,
            api_key=params.api_key,
            base_url=params.base_url,
            enable_web_search=params.enable_web_search,
        )
        return GenerateAnswerResult(answer=answer)

    def retrieve_and_answer_tool(
        params: RetrieveAndAnswerInput,
    ) -> RetrieveAndAnswerResult:
        """Retrieve context and generate an answer in one call."""
        context = retrieve_context(
            query=params.query,
            index_path=params.index_path,
            meta_path=params.meta_path,
            graphrag_dir=params.graphrag_dir,
            api_key=params.api_key,
            base_url=params.base_url,
            embedding_base_url=params.embedding_base_url,
            rerank_base_url=params.rerank_base_url,
            rerank_model=params.rerank_model,
            rerank_top_k=params.rerank_top_k,
            diversity=params.diversity,
            top_k_faiss=params.top_k_faiss,
            retrieval_method=params.retrieval_method,
            use_graphrag=params.use_graphrag,
            retriever_model=params.retriever_model,
            compressor_model=params.compressor_model,
            chat_model=params.chat_model or _resolve_response_model(params),
        )
        answer = generate_answer(
            query=params.query,
            context_text=context["context_text"],
            model=_resolve_response_model(params),
            system=params.system,
            api_key=params.api_key,
            base_url=params.base_url,
            enable_web_search=params.enable_web_search,
        )
        return RetrieveAndAnswerResult(
            answer=answer,
            context_text=context["context_text"],
            retrieved_context_text=context.get("retrieved_context_text", ""),
            rerank_chunks=context.get("rerank_chunks", []),
            sources=context["sources"],
            source_documents=context.get("source_documents", []),
            source_document_details=context.get("source_document_details", []),
            embedding_model=context.get("embedding_model", "text-embedding-3-small"),
            embedding_profile=context.get("embedding_profile", ""),
        )

    mcp.tool()(kb_build)
    mcp.tool()(kb_load)
    mcp.tool()(kb_append)
    mcp.tool()(retrieve_context_tool)
    mcp.tool()(generate_answer_tool)
    mcp.tool()(retrieve_and_answer_tool)
