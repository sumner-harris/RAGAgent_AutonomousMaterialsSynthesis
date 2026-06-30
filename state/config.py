import re

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import tiktoken

set_llm_cache(SQLiteCache(database_path=".cache.db"))

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
SFR_EMBEDDING_MODEL = "Salesforce/SFR-Embedding-Mistral"
DEFAULT_EMBEDDING_MODEL_OPTIONS = list(EMBEDDING_DIMENSIONS) + [
    SFR_EMBEDDING_MODEL,
]
DEFAULT_EMBEDDING_PROFILE = "default"
SFR_EMBEDDING_PROFILE = "sfr"
EMBEDDING_PROFILE_OPTIONS = [
    DEFAULT_EMBEDDING_PROFILE,
    SFR_EMBEDDING_PROFILE,
]
EMBEDDING_PROFILE_LABELS = {
    DEFAULT_EMBEDDING_PROFILE: "Standard",
    SFR_EMBEDDING_PROFILE: "SFR",
}
KB_EMBEDDING_BATCH_SIZE = 64

TOKENS_PER_CHUNK = 300
WORDS_PER_CHUNK_OVERLAP = int(TOKENS_PER_CHUNK / 5)  # ~20%

# Retrieval sizes
TOP_K_TEXT_FAISS = 50
TOP_K_TEXT_BM25 = 50
TOP_K_UPLOAD_FAISS = 25
TOP_K_GRAPH = 25
DEFAULT_RETRIEVAL_DIVERSITY = 0.7
DEFAULT_RERANK_TOP_K = 15
GRAPHRAG_CONCURRENT_REQUESTS_OPENAI = 25
GRAPHRAG_CONCURRENT_REQUESTS_LOCAL = 2

STREAM_DELAY = 0.08

TOKENIZER_NAME = "cl100k_base"
DEFAULT_RETRIEVAL_METHOD = "hybrid_multi_query_compression"
GRAPHRAG_GLOBAL_RETRIEVAL_METHOD = "graphrag_global"
GRAPHRAG_LOCAL_RETRIEVAL_METHOD = "graphrag_local"

RETRIEVAL_METHODS = {
    "faiss_mmr": {
        "label": "Vector (FAISS + MMR)",
        "description": "Vector-only retrieval from the FAISS index using Max Marginal Relevance.",
    },
    "bm25": {
        "label": "BM25 (keyword)",
        "description": "Keyword-based lexical retrieval without vector search or LLM expansion.",
    },
    "hybrid": {
        "label": "Hybrid (FAISS + BM25)",
        "description": "Combines vector and lexical retrieval without extra LLM query expansion.",
    },
    "hybrid_multi_query": {
        "label": "Hybrid + Multi-Query",
        "description": "Hybrid retrieval with LLM-generated query expansion.",
    },
    "hybrid_multi_query_compression": {
        "label": "Hybrid + Multi-Query + Compression",
        "description": "The previous default pipeline with hybrid retrieval, query expansion, and compression.",
    },
    GRAPHRAG_GLOBAL_RETRIEVAL_METHOD: {
        "label": "GraphRAG (Global)",
        "description": "Graph-only retrieval using GraphRAG community summaries for broad, high-level questions.",
    },
    GRAPHRAG_LOCAL_RETRIEVAL_METHOD: {
        "label": "GraphRAG (Local)",
        "description": "Graph-only retrieval using GraphRAG local entity and relationship context for focused questions.",
    },
}

LEGACY_RETRIEVAL_METHOD_ALIASES = {
    "graphrag": GRAPHRAG_GLOBAL_RETRIEVAL_METHOD,
}

VECTOR_RETRIEVAL_METHODS = {
    "faiss_mmr",
    "hybrid",
    "hybrid_multi_query",
    "hybrid_multi_query_compression",
}

BM25_RETRIEVAL_METHODS = {
    "bm25",
    "hybrid",
    "hybrid_multi_query",
    "hybrid_multi_query_compression",
}

MULTI_QUERY_RETRIEVAL_METHODS = {
    "hybrid_multi_query",
    "hybrid_multi_query_compression",
}

COMPRESSED_RETRIEVAL_METHODS = {
    "hybrid_multi_query_compression",
}

GRAPHRAG_RETRIEVAL_METHODS = {
    GRAPHRAG_GLOBAL_RETRIEVAL_METHOD,
    GRAPHRAG_LOCAL_RETRIEVAL_METHOD,
}


class FallbackTokenizer:
    def encode(self, text):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def decode(self, tokens):
        decoded = []
        punctuation = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
        closing = {"'", '"'}
        opening = {"(", "[", "{", "$", "#"}
        for token in tokens:
            token = str(token)
            if not decoded:
                decoded.append(token)
                continue
            if token in punctuation or token in closing:
                decoded[-1] += token
            elif decoded[-1] in opening:
                decoded[-1] += token
            else:
                decoded.append(token)
        return " ".join(decoded)


TOKENIZER_WARNING = None
try:
    ENC = tiktoken.get_encoding(TOKENIZER_NAME)
except Exception as exc:
    ENC = FallbackTokenizer()
    TOKENIZER_WARNING = (
        "Offline tokenizer fallback is active because the OpenAI tokenizer files "
        f"could not be loaded: {exc}"
    )
