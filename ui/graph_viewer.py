import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from RAG.services.graph_view_service import (
    GraphViewLoadError,
    build_graphviz_dot,
    build_interactive_graph_payload,
    load_graph_view_data,
)


def graph_viewer_panel():
    """Render a GraphRAG knowledge-graph viewer."""
    st.header("Knowledge Graph Viewer")

    loaded_workspace = st.session_state.get("graphrag") or ""
    current_workspace = st.session_state.get("graph_viewer_workspace", "")
    if loaded_workspace and not current_workspace:
        st.session_state.graph_viewer_workspace = loaded_workspace

    workspace = st.text_input(
        "GraphRAG workspace directory",
        key="graph_viewer_workspace",
        help="Use the GraphRAG workspace root or its output directory.",
    ).strip()

    if not workspace:
        st.info(
            "Load a knowledge base with a GraphRAG workspace, or paste a GraphRAG "
            "workspace path here to open the viewer."
        )
        return

    try:
        graph_payload = load_graph_view_data(graphrag_dir=workspace)
    except GraphViewLoadError as exc:
        st.error(str(exc))
        return

    _render_summary(graph_payload)

    communities = graph_payload.get("communities", [])
    community_options = [""] + [community["id"] for community in communities]
    community_labels = {
        "": "All communities",
        **{
            community["id"]: _community_label(community)
            for community in communities
        },
    }
    if st.session_state.get("graph_viewer_selected_community") not in community_options:
        st.session_state.graph_viewer_selected_community = ""

    controls_left, controls_right = st.columns(2)
    with controls_left:
        selected_community_id = st.selectbox(
            "Community filter",
            community_options,
            key="graph_viewer_selected_community",
            format_func=lambda value: community_labels.get(value, value),
        )
        max_nodes = st.slider(
            "Max entities",
            min_value=10,
            max_value=200,
            key="graph_viewer_max_nodes",
            step=10,
        )
    with controls_right:
        search_text = st.text_input(
            "Search entities",
            key="graph_viewer_search_text",
            help="Filter by entity title, description, type, or community ID.",
        )
        max_edges = st.slider(
            "Max relationships",
            min_value=10,
            max_value=400,
            key="graph_viewer_max_edges",
            step=10,
        )

    view_mode = st.radio(
        "Viewer mode",
        options=("Interactive", "Graphviz"),
        horizontal=True,
        key="graph_viewer_mode",
    )

    if view_mode == "Interactive":
        interactive_payload = build_interactive_graph_payload(
            graph_payload=graph_payload,
            selected_community_id=selected_community_id,
            search_text=search_text,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not interactive_payload["nodes"]:
            st.info("No graph nodes matched the current filters.")
            return
        components.html(
            _build_interactive_graph_html(interactive_payload),
            height=760,
            scrolling=False,
        )
        visible_nodes = interactive_payload["nodes"]
        visible_edges = interactive_payload["edges"]
    else:
        dot_graph, visible_nodes, visible_edges = build_graphviz_dot(
            graph_payload=graph_payload,
            selected_community_id=selected_community_id,
            search_text=search_text,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not visible_nodes:
            st.info("No graph nodes matched the current filters.")
            return
        st.graphviz_chart(dot_graph, use_container_width=True)

    with st.expander("Visible graph details"):
        st.caption(
            f"Showing {len(visible_nodes)} entities and {len(visible_edges)} relationships."
        )
        st.dataframe(_nodes_frame(visible_nodes), use_container_width=True)
        if visible_edges:
            st.dataframe(_edges_frame(visible_edges), use_container_width=True)


def _render_summary(graph_payload: dict):
    """Render high-level GraphRAG graph metrics."""
    workspace_dir = graph_payload.get("workspace_dir", "")
    st.caption(f"Workspace: {workspace_dir}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Entities", int(graph_payload.get("node_count", 0)))
    col2.metric("Relationships", int(graph_payload.get("edge_count", 0)))
    col3.metric("Communities", int(graph_payload.get("community_count", 0)))


def _community_label(community: dict) -> str:
    """Format a community option for display."""
    title = str(community.get("title", "")).strip() or str(community.get("id", ""))
    community_id = str(community.get("id", "")).strip()
    level = str(community.get("level", "")).strip()
    if level:
        return f"{title} (ID {community_id}, level {level})"
    return f"{title} (ID {community_id})"


def _nodes_frame(nodes: list[dict]) -> pd.DataFrame:
    """Build a display dataframe for visible graph nodes."""
    return pd.DataFrame(
        [
            {
                "title": node.get("title", ""),
                "short_id": node.get("short_id", ""),
                "type": node.get("entity_type", ""),
                "degree": int(node.get("degree", 0)),
                "communities": ", ".join(node.get("community_ids", [])),
                "description": node.get("description", ""),
            }
            for node in nodes
        ]
    )


def _edges_frame(edges: list[dict]) -> pd.DataFrame:
    """Build a display dataframe for visible graph edges."""
    return pd.DataFrame(
        [
            {
                "source": edge.get("source_title", ""),
                "target": edge.get("target_title", ""),
                "weight": float(edge.get("weight", 0.0)),
                "rank": float(edge.get("rank", 0.0)),
                "description": edge.get("description", ""),
            }
            for edge in edges
        ]
    )


def _build_interactive_graph_html(graph_payload: dict) -> str:
    """Build a self-contained SVG-based graph viewer."""
    serialized = json.dumps(graph_payload)
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body {{
        margin: 0;
        font-family: Arial, sans-serif;
        background: #f8fafc;
        color: #0f172a;
      }}
      .viewer {{
        padding: 12px;
      }}
      .frame {{
        position: relative;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        overflow: hidden;
      }}
      svg {{
        width: 100%;
        height: 680px;
        display: block;
      }}
      .tooltip {{
        position: absolute;
        pointer-events: none;
        max-width: 320px;
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
        font-size: 12px;
        line-height: 1.45;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.25);
        opacity: 0;
        transition: opacity 120ms ease-in-out;
      }}
      .legend {{
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 10px;
        font-size: 12px;
        color: #475569;
      }}
      .legend strong {{
        color: #0f172a;
      }}
    </style>
  </head>
  <body>
    <div class="viewer">
      <div class="frame">
        <svg id="graph" viewBox="0 0 1040 680" aria-label="Knowledge graph viewer"></svg>
        <div id="tooltip" class="tooltip"></div>
      </div>
      <div class="legend">
        <div><strong>Node size</strong>: higher degree</div>
        <div><strong>Edge width</strong>: higher GraphRAG relationship weight</div>
        <div><strong>Hover</strong>: entity details</div>
      </div>
    </div>
    <script>
      const payload = {serialized};
      const svg = document.getElementById("graph");
      const tooltip = document.getElementById("tooltip");
      const namespace = "http://www.w3.org/2000/svg";

      function escapeHtml(value) {{
        return String(value || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}

      function makeSvg(tagName, attrs) {{
        const element = document.createElementNS(namespace, tagName);
        Object.entries(attrs || {{}}).forEach(([key, value]) => {{
          element.setAttribute(key, String(value));
        }});
        return element;
      }}

      function showTooltip(event, node) {{
        const communities = (node.community_ids || []).join(", ") || "None";
        const description = node.description || "No description available.";
        tooltip.innerHTML =
          "<strong>" + escapeHtml(node.title) + "</strong><br/>" +
          "ID: " + escapeHtml(node.short_id || node.id) + "<br/>" +
          "Type: " + escapeHtml(node.entity_type || "Unknown") + "<br/>" +
          "Degree: " + escapeHtml(node.degree) + "<br/>" +
          "Communities: " + escapeHtml(communities) + "<br/><br/>" +
          escapeHtml(description);
        tooltip.style.left = (event.offsetX + 18) + "px";
        tooltip.style.top = (event.offsetY + 18) + "px";
        tooltip.style.opacity = "1";
      }}

      function hideTooltip() {{
        tooltip.style.opacity = "0";
      }}

      const nodes = payload.nodes || [];
      const edges = payload.edges || [];
      const nodeById = Object.fromEntries(nodes.map((node) => [node.id, node]));

      edges.forEach((edge) => {{
        const source = nodeById[edge.source_id];
        const target = nodeById[edge.target_id];
        if (!source || !target) {{
          return;
        }}
        const line = makeSvg("line", {{
          x1: source.x,
          y1: source.y,
          x2: target.x,
          y2: target.y,
          stroke: "#94a3b8",
          "stroke-width": Math.max(1, Math.min(5, 1 + Number(edge.weight || 0))),
          "stroke-opacity": 0.75
        }});
        svg.appendChild(line);
      }});

      nodes.forEach((node) => {{
        const group = makeSvg("g");
        const circle = makeSvg("circle", {{
          cx: node.x,
          cy: node.y,
          r: node.radius || 18,
          fill: "#dbeafe",
          stroke: "#2563eb",
          "stroke-width": 2
        }});
        const label = makeSvg("text", {{
          x: node.x,
          y: Number(node.y) + Number(node.radius || 18) + 16,
          "font-size": 12,
          "font-family": "Arial, sans-serif",
          "text-anchor": "middle",
          fill: "#0f172a"
        }});
        label.textContent = node.title;
        group.appendChild(circle);
        group.appendChild(label);
        group.addEventListener("mousemove", (event) => showTooltip(event, node));
        group.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(group);
      }});
    </script>
  </body>
</html>
"""
