import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


class GraphViewLoadError(RuntimeError):
    """Raised when GraphRAG viewer data cannot be loaded."""


def load_graph_view_data(*, graphrag_dir: str) -> dict[str, Any]:
    """Load entity and relationship data from a GraphRAG workspace.

    Args:
        graphrag_dir: GraphRAG workspace root or its `output` directory.

    Returns:
        Dictionary with stable keys for nodes, edges, communities, and counts.

    Raises:
        GraphViewLoadError: When the workspace path or GraphRAG artifacts are invalid.
    """
    output_dir = _resolve_output_dir(graphrag_dir)
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    communities_path = output_dir / "communities.parquet"

    missing = [
        str(path)
        for path in (entities_path, relationships_path, communities_path)
        if not path.exists()
    ]
    if missing:
        raise GraphViewLoadError(
            "Missing GraphRAG viewer artifacts: " + ", ".join(missing)
        )

    entities_df = _read_parquet(entities_path)
    relationships_df = _read_parquet(relationships_path)
    communities_df = _read_parquet(communities_path)

    nodes_by_id: dict[str, dict[str, Any]] = {}
    nodes_by_title: dict[str, dict[str, Any]] = {}

    for row in entities_df.to_dict(orient="records"):
        title = _coerce_text(row.get("title"))
        short_id = _coerce_text(row.get("human_readable_id"))
        node_id = _coerce_text(row.get("id")) or title or short_id
        if not node_id:
            continue
        if not title:
            title = short_id or node_id

        node = {
            "id": node_id,
            "title": title,
            "short_id": short_id or node_id,
            "entity_type": _coerce_text(row.get("type")),
            "description": _coerce_text(row.get("description")),
            "community_ids": _coerce_list_of_text(row.get("community_ids")),
            "degree": 0,
        }
        nodes_by_id[node_id] = node
        nodes_by_title[title] = node

    edges: list[dict[str, Any]] = []
    for row in relationships_df.to_dict(orient="records"):
        source_title = _coerce_text(row.get("source"))
        target_title = _coerce_text(row.get("target"))
        if not source_title or not target_title:
            continue

        source_node = nodes_by_title.get(source_title)
        if source_node is None:
            source_node = _build_placeholder_node(source_title)
            nodes_by_id[source_node["id"]] = source_node
            nodes_by_title[source_title] = source_node

        target_node = nodes_by_title.get(target_title)
        if target_node is None:
            target_node = _build_placeholder_node(target_title)
            nodes_by_id[target_node["id"]] = target_node
            nodes_by_title[target_title] = target_node

        source_node["degree"] = int(source_node.get("degree", 0)) + 1
        target_node["degree"] = int(target_node.get("degree", 0)) + 1
        edges.append(
            {
                "id": _coerce_text(row.get("id"))
                or _coerce_text(row.get("human_readable_id"))
                or f"{source_node['id']}->{target_node['id']}",
                "source_id": source_node["id"],
                "target_id": target_node["id"],
                "source_title": source_node["title"],
                "target_title": target_node["title"],
                "weight": _coerce_float(row.get("weight"), default=1.0),
                "rank": _coerce_float(row.get("rank"), default=0.0),
                "description": _coerce_text(row.get("description")),
            }
        )

    communities = _build_communities(communities_df)
    nodes = sorted(
        nodes_by_id.values(),
        key=lambda item: (
            -int(item.get("degree", 0)),
            str(item.get("title", "")).lower(),
            str(item.get("id", "")).lower(),
        ),
    )
    edges.sort(
        key=lambda item: (
            -float(item.get("weight", 0.0)),
            -float(item.get("rank", 0.0)),
            str(item.get("source_title", "")).lower(),
            str(item.get("target_title", "")).lower(),
        )
    )

    return {
        "workspace_dir": str(output_dir.parent),
        "output_dir": str(output_dir),
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "community_count": len(communities),
    }


def build_graphviz_dot(
    *,
    graph_payload: dict[str, Any],
    selected_community_id: str | None,
    search_text: str | None,
    max_nodes: int,
    max_edges: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a filtered Graphviz DOT graph for a GraphRAG workspace."""
    filtered_nodes, filtered_edges = _filter_graph_payload(
        graph_payload=graph_payload,
        selected_community_id=selected_community_id,
        search_text=search_text,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )

    lines = [
        "digraph knowledge_graph {",
        '  graph [overlap=false, splines=true, rankdir=LR];',
        '  node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#60a5fa"];',
        '  edge [color="#94a3b8"];',
    ]
    for node in filtered_nodes:
        label = _escape_dot_label(
            f"{node['title']}\\nID={node['short_id']}\\ndegree={node['degree']}"
        )
        lines.append(f'  "{node["id"]}" [label="{label}"];')
    for edge in filtered_edges:
        pen_width = max(1.0, min(4.0, 1.0 + float(edge.get("weight", 0.0))))
        lines.append(
            f'  "{edge["source_id"]}" -> "{edge["target_id"]}" '
            f'[label="{float(edge.get("weight", 0.0)):.2f}", penwidth={pen_width:.2f}];'
        )
    lines.append("}")
    return "\n".join(lines), filtered_nodes, filtered_edges


def build_interactive_graph_payload(
    *,
    graph_payload: dict[str, Any],
    selected_community_id: str | None,
    search_text: str | None,
    max_nodes: int,
    max_edges: int,
) -> dict[str, Any]:
    """Build a filtered graph payload with deterministic layout positions."""
    filtered_nodes, filtered_edges = _filter_graph_payload(
        graph_payload=graph_payload,
        selected_community_id=selected_community_id,
        search_text=search_text,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    return {
        "nodes": _layout_nodes(nodes=filtered_nodes),
        "edges": filtered_edges,
    }


def _resolve_output_dir(graphrag_dir: str) -> Path:
    """Resolve a GraphRAG workspace root or output directory to `output`."""
    raw_workspace = str(graphrag_dir or "").strip()
    if not raw_workspace:
        raise GraphViewLoadError("No GraphRAG workspace directory was provided.")
    workspace = Path(raw_workspace).expanduser()
    if not workspace.exists():
        raise GraphViewLoadError(
            f"GraphRAG workspace directory does not exist: {workspace}"
        )
    if workspace.is_dir() and workspace.name.lower() == "output":
        return workspace
    output_dir = workspace / "output"
    if output_dir.is_dir():
        return output_dir
    raise GraphViewLoadError(
        f"GraphRAG output directory not found under workspace: {workspace}"
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read one parquet file with actionable error messages."""
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise GraphViewLoadError(
            "Parquet support is unavailable while loading the graph viewer. "
            "Install a parquet engine such as pyarrow."
        ) from exc
    except Exception as exc:
        raise GraphViewLoadError(f"Failed to read GraphRAG artifact at {path}.") from exc


def _build_communities(communities_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize GraphRAG community rows for viewer filters."""
    communities = []
    for row in communities_df.to_dict(orient="records"):
        community_id = _coerce_text(row.get("id")) or _coerce_text(
            row.get("human_readable_id")
        )
        if not community_id:
            continue
        communities.append(
            {
                "id": community_id,
                "title": _coerce_text(row.get("title")) or community_id,
                "level": _coerce_text(row.get("level")),
                "short_id": _coerce_text(row.get("human_readable_id"))
                or community_id,
            }
        )
    communities.sort(key=lambda item: (item["title"].lower(), item["id"].lower()))
    return communities


def _filter_graph_payload(
    *,
    graph_payload: dict[str, Any],
    selected_community_id: str | None,
    search_text: str | None,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter nodes and edges for community, search text, and size limits."""
    community_filter = str(selected_community_id or "").strip()
    search_filter = str(search_text or "").strip().lower()

    filtered_nodes = []
    for node in graph_payload.get("nodes", []):
        community_ids = [str(item).strip() for item in node.get("community_ids", [])]
        if community_filter and community_filter not in community_ids:
            continue
        if search_filter:
            haystack = " ".join(
                [
                    str(node.get("title", "")),
                    str(node.get("short_id", "")),
                    str(node.get("entity_type", "")),
                    str(node.get("description", "")),
                    " ".join(community_ids),
                ]
            ).lower()
            if search_filter not in haystack:
                continue
        filtered_nodes.append(dict(node))

    limited_nodes = filtered_nodes[: max(1, int(max_nodes))]
    allowed_node_ids = {str(node.get("id", "")).strip() for node in limited_nodes}

    filtered_edges = [
        dict(edge)
        for edge in graph_payload.get("edges", [])
        if str(edge.get("source_id", "")).strip() in allowed_node_ids
        and str(edge.get("target_id", "")).strip() in allowed_node_ids
    ]
    limited_edges = filtered_edges[: max(1, int(max_edges))]
    return limited_nodes, limited_edges


def _build_placeholder_node(title: str) -> dict[str, Any]:
    """Create a fallback node for relationships missing from entities."""
    return {
        "id": title,
        "title": title,
        "short_id": title,
        "entity_type": "",
        "description": "",
        "community_ids": [],
        "degree": 0,
    }


def _coerce_text(value: Any) -> str:
    """Convert viewer values to a trimmed display string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_float(value: Any, *, default: float) -> float:
    """Convert viewer numeric values with a default fallback."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_list_of_text(value: Any) -> list[str]:
    """Normalize a GraphRAG list-like field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            items = [part.strip() for part in stripped.split(",")]
        else:
            items = parsed if isinstance(parsed, list) else [parsed]
    else:
        items = [value]

    normalized = []
    for item in items:
        text = _coerce_text(item)
        if text:
            normalized.append(text)
    return normalized


def _escape_dot_label(value: str) -> str:
    """Escape Graphviz label text."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _layout_nodes(*, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign deterministic positions for interactive graph rendering."""
    if not nodes:
        return []
    count = len(nodes)
    center_x = 520.0
    center_y = 320.0
    base_radius = max(150.0, min(340.0, 85.0 + (count * 12.0)))
    laid_out = []
    ordered_nodes = sorted(nodes, key=lambda item: str(item.get("title", "")).lower())

    for index, raw_node in enumerate(ordered_nodes):
        node = dict(raw_node)
        angle = (2.0 * math.pi * index) / max(count, 1)
        orbit = base_radius + ((index % 3) * 30.0)
        degree = int(node.get("degree", 0))
        node["x"] = center_x + (orbit * math.cos(angle))
        node["y"] = center_y + (orbit * math.sin(angle))
        node["radius"] = max(18.0, min(34.0, 16.0 + (2.0 * degree)))
        laid_out.append(node)
    return laid_out
