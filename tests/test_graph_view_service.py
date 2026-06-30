from pathlib import Path

import pandas as pd
import pytest

from RAG.services.graph_view_service import (
    GraphViewLoadError,
    build_graphviz_dot,
    build_interactive_graph_payload,
    load_graph_view_data,
)


def _write_placeholder(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def test_load_graph_view_data_reads_graphrag_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "graphrag"
    output_dir = workspace / "output"
    for name in ("entities.parquet", "relationships.parquet", "communities.parquet"):
        _write_placeholder(output_dir / name)

    entities_df = pd.DataFrame(
        [
            {
                "id": "entity-1",
                "human_readable_id": "1",
                "title": "Catalyst A",
                "type": "material",
                "description": "Alpha catalyst",
                "community_ids": ["10"],
            },
            {
                "id": "entity-2",
                "human_readable_id": "2",
                "title": "Solvent B",
                "type": "solvent",
                "description": "Polar solvent",
                "community_ids": ["10", "11"],
            },
            {
                "id": "entity-3",
                "human_readable_id": "3",
                "title": "Method C",
                "type": "process",
                "description": "Synthesis route",
                "community_ids": ["11"],
            },
        ]
    )
    relationships_df = pd.DataFrame(
        [
            {
                "human_readable_id": "101",
                "source": "Catalyst A",
                "target": "Solvent B",
                "weight": 0.9,
                "rank": 5,
                "description": "paired with",
            },
            {
                "human_readable_id": "102",
                "source": "Solvent B",
                "target": "Method C",
                "weight": 0.4,
                "rank": 2,
                "description": "used in",
            },
        ]
    )
    communities_df = pd.DataFrame(
        [
            {"id": "10", "human_readable_id": "10", "title": "Catalysis", "level": 1},
            {"id": "11", "human_readable_id": "11", "title": "Processing", "level": 2},
        ]
    )

    def _fake_read_parquet(path):
        name = Path(path).name
        if name == "entities.parquet":
            return entities_df
        if name == "relationships.parquet":
            return relationships_df
        if name == "communities.parquet":
            return communities_df
        raise AssertionError(f"Unexpected parquet path: {path}")

    monkeypatch.setattr("RAG.services.graph_view_service.pd.read_parquet", _fake_read_parquet)

    payload = load_graph_view_data(graphrag_dir=str(workspace))

    assert payload["workspace_dir"] == str(workspace)
    assert payload["node_count"] == 3
    assert payload["edge_count"] == 2
    assert payload["community_count"] == 2
    assert payload["nodes"][0]["title"] == "Solvent B"
    assert payload["nodes"][0]["degree"] == 2
    assert payload["communities"][0]["title"] == "Catalysis"


def test_graph_view_filters_nodes_edges_and_layout(monkeypatch, tmp_path):
    workspace = tmp_path / "graphrag"
    output_dir = workspace / "output"
    for name in ("entities.parquet", "relationships.parquet", "communities.parquet"):
        _write_placeholder(output_dir / name)

    monkeypatch.setattr(
        "RAG.services.graph_view_service.pd.read_parquet",
        lambda path: {
            "entities.parquet": pd.DataFrame(
                [
                    {
                        "id": "entity-1",
                        "human_readable_id": "1",
                        "title": "Catalyst A",
                        "type": "material",
                        "description": "Alpha catalyst",
                        "community_ids": ["10"],
                    },
                    {
                        "id": "entity-2",
                        "human_readable_id": "2",
                        "title": "Solvent B",
                        "type": "solvent",
                        "description": "Polar solvent",
                        "community_ids": ["10", "11"],
                    },
                    {
                        "id": "entity-3",
                        "human_readable_id": "3",
                        "title": "Method C",
                        "type": "process",
                        "description": "Synthesis route",
                        "community_ids": ["11"],
                    },
                ]
            ),
            "relationships.parquet": pd.DataFrame(
                [
                    {
                        "human_readable_id": "101",
                        "source": "Catalyst A",
                        "target": "Solvent B",
                        "weight": 0.9,
                        "rank": 5,
                        "description": "paired with",
                    },
                    {
                        "human_readable_id": "102",
                        "source": "Solvent B",
                        "target": "Method C",
                        "weight": 0.4,
                        "rank": 2,
                        "description": "used in",
                    },
                ]
            ),
            "communities.parquet": pd.DataFrame(
                [
                    {
                        "id": "10",
                        "human_readable_id": "10",
                        "title": "Catalysis",
                        "level": 1,
                    },
                    {
                        "id": "11",
                        "human_readable_id": "11",
                        "title": "Processing",
                        "level": 2,
                    },
                ]
            ),
        }[Path(path).name],
    )

    payload = load_graph_view_data(graphrag_dir=str(workspace))
    dot, visible_nodes, visible_edges = build_graphviz_dot(
        graph_payload=payload,
        selected_community_id="10",
        search_text="",
        max_nodes=10,
        max_edges=10,
    )
    interactive = build_interactive_graph_payload(
        graph_payload=payload,
        selected_community_id="10",
        search_text="",
        max_nodes=10,
        max_edges=10,
    )

    assert len(visible_nodes) == 2
    assert len(visible_edges) == 1
    assert "Catalyst A" in dot
    assert "Solvent B" in dot
    assert len(interactive["nodes"]) == 2
    assert interactive["nodes"][0]["x"] != interactive["nodes"][1]["x"]
    assert interactive["edges"][0]["source_title"] == "Catalyst A"


def test_load_graph_view_data_requires_valid_workspace(tmp_path):
    with pytest.raises(GraphViewLoadError, match="does not exist"):
        load_graph_view_data(graphrag_dir=str(tmp_path / "missing"))
