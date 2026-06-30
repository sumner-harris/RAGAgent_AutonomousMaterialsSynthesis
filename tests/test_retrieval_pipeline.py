from RAG.retrieval.pipeline import resolve_retrieval_method
from state.config import GRAPHRAG_GLOBAL_RETRIEVAL_METHOD


def test_resolve_retrieval_method_maps_legacy_graphrag_alias():
    assert resolve_retrieval_method("graphrag") == GRAPHRAG_GLOBAL_RETRIEVAL_METHOD


def test_resolve_retrieval_method_defaults_use_graphrag_to_global():
    assert (
        resolve_retrieval_method(None, use_graphrag=True)
        == GRAPHRAG_GLOBAL_RETRIEVAL_METHOD
    )
