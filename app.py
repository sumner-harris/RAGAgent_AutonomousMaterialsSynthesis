# app.py
import streamlit as st

from state.session import init_session
from state.config import TOKENIZER_WARNING

from ui.API_setup import API_entry
from ui.graph_viewer import graph_viewer_panel
from ui.kb_setup import kb_setup
from ui.qa_panel import qa_panel

from ingestion.embedding_profiles import create_embedding_function

init_session()

# --------------------------------
# App layout
# --------------------------------
st.markdown("## 📄 RAG Agent — Autonomous Synthesis")
if TOKENIZER_WARNING:
    st.warning(
        "Tokenizer files could not be fetched, so the app is using an offline "
        "fallback for chunk sizing. Retrieval still works, but token counts and "
        "chunk boundaries may be approximate."
    )
st.divider()

# --------------------------------
# API entry / auth
# --------------------------------
API_entry()
client = st.session_state.client
st.divider()

# --------------------------------
# Knowledge base setup
# --------------------------------
embeddings_obj = None
if st.session_state.api_verified:
    embeddings_obj = create_embedding_function(
        model=st.session_state.get("embedding_model", "text-embedding-3-small"),
        profile_name=st.session_state.get("embedding_profile"),
        api_key=st.session_state.api_key,
        base_url=st.session_state.embedding_api_base_url,
        fallback_to_env=False,
    )

kb_setup(
    client=client,
    embeddings=embeddings_obj,
)

st.divider()

# --------------------------------
# Q&A + Graph viewer
# --------------------------------
qa_tab, graph_tab = st.tabs(["Ask a Question", "Knowledge Graph Viewer"])
with qa_tab:
    qa_panel(client=client)
with graph_tab:
    graph_viewer_panel()
