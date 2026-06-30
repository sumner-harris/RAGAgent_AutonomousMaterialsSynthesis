import base64
import io
import mimetypes

import fitz
import pandas as pd
import streamlit as st

from ingestion.embedding_profiles import (
    create_embedding_function,
    format_embedding_documents,
)
from RAG.openai_compat import make_openai_client
from state.config import (
    ENC,
    KB_EMBEDDING_BATCH_SIZE,
    TOKENS_PER_CHUNK,
    WORDS_PER_CHUNK_OVERLAP,
)
from state.schemas import ChunkMetadata, UploadBundle
from ingestion.chunkers import chunk_text2
from ingestion.cleaners import remove_junk_sections, remove_junk_lines
from ingestion.faiss_store import build_faiss_from_embeddings

# ------------------------------
# PDF
# ------------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# ------------------------------
# Tables (CSV / XLSX)
# ------------------------------
def _parse_table_file(file_bytes, filename, max_rows=50, max_chars=20000):
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        return f"[PARSE-ERROR {filename}: {e}]"

    buf = []
    buf.append(f"TABLE: {filename}")
    buf.append("COLUMNS: " + ", ".join(map(str, df.columns.tolist())))
    head = df.head(max_rows)
    buf.append("HEAD:")
    buf.append(head.to_csv(index=False))

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]..."

    return text

# ------------------------------
# Images
# ------------------------------
def _to_data_url(file_bytes, mime_type="image/png"):
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _is_image(filename):
    return filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff")
    )


def _is_text_like(filename):
    name = filename.lower()
    return name.endswith((".pdf", ".txt", ".csv", ".xlsx"))


def _extract_text_from_upload(name, data):
    if name.lower().endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    if name.lower().endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.lower().endswith((".csv", ".xlsx")):
        return _parse_table_file(data, name)
    return ""


# ------------------------------
# Uploaded files
# ------------------------------
def build_upload_bundle(
    uploaded_files,
    client,
    embedding_model,
    embedding_profile,
    dimension,
    callbacks=None,
    api_key=None,
    base_url=None,
):
    """
    Build in-memory FAISS for uploaded *text-like* files and collect images.
    Returns: (upload_db, text_meta, images)
      - upload_db: FAISS store for uploaded text chunks (or None if none)
      - text_meta: list of dicts for chunks
      - images: list of {"name": str, "data_url": str}
    """
    text_chunks = []
    text_meta = []
    images = []

    if not uploaded_files:
        return None, [], []

    for uf in uploaded_files:
        name = uf.name
        mime = uf.type or mimetypes.guess_type(name)[0] or ""
        data = uf.getvalue()  # bytes

        # Images
        if _is_image(name):
            data_url = _to_data_url(data, mime or "image/png")
            images.append({"name": name, "data_url": data_url})
            continue

        # Text-like docs
        if not _is_text_like(name):
            _call(
                _get_cb(callbacks, "warning"),
                (
                    f"⚠️ Unsupported file type: {name} "
                    f"(supported: pdf/txt/csv/xlsx + images)"
                ),
            )
            continue

        try:
            text = _extract_text_from_upload(name, data)
        except Exception as exc:
            _call(_get_cb(callbacks, "warning"), f"⚠️ Could not read {name}: {exc}")
            continue

        # Clean + chunk
        text = remove_junk_sections(text)
        text = remove_junk_lines(text)
        chunks = chunk_text2(
            text,
            max_tokens=TOKENS_PER_CHUNK,
            tokenizer=ENC,
            overlap=WORDS_PER_CHUNK_OVERLAP,
        )
        for i, ch in enumerate(chunks):
            text_chunks.append(ch)
            meta = ChunkMetadata(
                source=f"uploaded/{name}",
                chunk_id=i,
                text=ch,
                embedding_model=embedding_model,
                embedding_profile=embedding_profile,
                embedding_dimension=dimension,
            )
            text_meta.append(meta.model_dump())

    # Validate upload bundle schema
    bundle = UploadBundle.model_validate(
        {
            "text_meta": text_meta,
            "images": images,
        }
    )
    bundle_data = bundle.model_dump()
    text_meta = bundle_data["text_meta"]
    images = bundle_data["images"]

    # Build FAISS for uploaded text
    upload_db = None
    if text_chunks:
        embedding_client = make_openai_client(
            api_key=api_key or st.session_state.api_key,
            base_url=base_url,
            fallback_to_env=False,
        )
        embs = []
        for i in range(0, len(text_chunks), KB_EMBEDDING_BATCH_SIZE):
            batch = text_chunks[i : i + KB_EMBEDDING_BATCH_SIZE]
            resp = embedding_client.embeddings.create(
                input=format_embedding_documents(
                    model_name=embedding_model,
                    texts=batch,
                    profile_name=embedding_profile,
                ),
                model=embedding_model,
            )
            embs.extend([d.embedding for d in resp.data])
        if embs and not dimension:
            dimension = len(embs[0])
            for meta in text_meta:
                meta["embedding_dimension"] = dimension

        embedding_fn = create_embedding_function(
            model=embedding_model,
            profile_name=embedding_profile,
            api_key=api_key or st.session_state.api_key,
            base_url=base_url,
            fallback_to_env=False,
        )
        _, upload_db, _ = build_faiss_from_embeddings(
            embs,
            text_meta,
            embedding_fn,
            dimension,
        )

    return upload_db, text_meta, images


def process_uploads_for_session(
    uploaded_files,
    client,
    embedding_model,
    embedding_profile,
    dimension,
    callbacks=None,
    api_key=None,
    base_url=None,
):
    """Build uploads and store results in Streamlit session_state.

    Returns: (num_text_chunks, num_images)
    """
    if not uploaded_files:
        st.session_state.upload_db = None
        st.session_state.upload_meta = []
        st.session_state.upload_images = []
        return 0, 0

    upload_db, text_meta, images = build_upload_bundle(
        uploaded_files=uploaded_files,
        client=client,
        embedding_model=embedding_model,
        embedding_profile=embedding_profile,
        dimension=dimension,
        callbacks=callbacks,
        api_key=api_key,
        base_url=base_url,
    )

    st.session_state.upload_db = upload_db
    st.session_state.upload_meta = text_meta
    st.session_state.upload_images = images

    return len(text_meta), len(images)
