from typing import List, Optional

from pydantic import BaseModel, Field
from state.config import DEFAULT_RETRIEVAL_METHOD

# ------------------------------
# Pydantic schemas for object/type consistency
# ------------------------------
class ChunkMetadata(BaseModel):
    source: str
    chunk_id: int
    text: str
    embedding_model: str
    embedding_profile: Optional[str] = None
    embedding_dimension: Optional[int] = None


class UploadImage(BaseModel):
    name: str
    data_url: str
    mime: Optional[str] = None


class UploadBundle(BaseModel):
    text_meta: List[ChunkMetadata] = Field(default_factory=list)
    images: List[UploadImage] = Field(default_factory=list)


class KBBuildRequest(BaseModel):
    pdf_dir: str
    index_path: str
    meta_path: str
    graphrag_dir: str = ""
    embedding_model: str
    embedding_profile: str = ""


class KBLoadRequest(BaseModel):
    index_path: str
    meta_path: str
    graphrag_dir: str = ""


class KBAppendRequest(BaseModel):
    index_path: str
    meta_path: str
    append_folder: str
    graphrag_dir: str = ""


class RetrievalRequest(BaseModel):
    query: str
    use_uploads: bool = True
    use_graphrag: bool = True
    retrieval_method: str = DEFAULT_RETRIEVAL_METHOD
    top_k_faiss: Optional[int] = None
    top_k_uploads: Optional[int] = None
    diversity: Optional[float] = None
