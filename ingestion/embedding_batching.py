from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ingestion.embedding_profiles import format_embedding_documents
from state.config import KB_EMBEDDING_BATCH_SIZE


@dataclass(frozen=True)
class ChunkRecord:
    """Represent one chunk that should be embedded."""

    source: str
    chunk_id: int
    text: str


def build_chunk_records(*, chunks_by_file: dict[str, list[str]]) -> list[ChunkRecord]:
    """Flatten grouped chunk text into stable per-chunk records.

    Args:
        chunks_by_file: Mapping of source filename to chunk text list.

    Returns:
        Ordered chunk records preserving file and chunk order.
    """
    records = []
    for source, chunks in chunks_by_file.items():
        for chunk_id, text in enumerate(chunks):
            records.append(ChunkRecord(source=source, chunk_id=chunk_id, text=text))
    return records


def _request_embeddings(
    *,
    embedding_client,
    model_name: str,
    profile_name: str | None,
    chunk_records: Sequence[ChunkRecord],
) -> list[list[float]]:
    """Send one embedding request and validate the response size."""
    response = embedding_client.embeddings.create(
        input=format_embedding_documents(
            model_name=model_name,
            texts=[record.text for record in chunk_records],
            profile_name=profile_name,
        ),
        model=model_name,
    )
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(chunk_records):
        raise ValueError(
            "Embedding response size mismatch. "
            f"Expected {len(chunk_records)} vectors, received {len(embeddings)}."
        )
    return embeddings


def embed_chunk_records(
    *,
    chunk_records: Sequence[ChunkRecord],
    embedding_client,
    model_name: str,
    profile_name: str | None,
    warning_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    batch_size: int = KB_EMBEDDING_BATCH_SIZE,
) -> list[tuple[ChunkRecord, list[float]]]:
    """Embed chunk records in batches and fall back to singles when needed.

    Args:
        chunk_records: Ordered chunk records to embed.
        embedding_client: OpenAI-compatible embeddings client.
        model_name: Embedding model identifier for the target endpoint.
        profile_name: Formatting profile applied before embedding.
        warning_callback: Optional callback for chunk-level failures.
        progress_callback: Optional callback receiving completed and total counts.
        batch_size: Maximum number of chunks to send in one request.

    Returns:
        Successful chunk/embedding pairs in input order.
    """
    if batch_size < 1:
        raise ValueError("Embedding batch size must be at least 1.")

    embedded_records = []
    total = len(chunk_records)
    completed = 0

    for start in range(0, total, batch_size):
        batch_records = list(chunk_records[start : start + batch_size])
        try:
            batch_embeddings = _request_embeddings(
                embedding_client=embedding_client,
                model_name=model_name,
                profile_name=profile_name,
                chunk_records=batch_records,
            )
            for record, embedding in zip(batch_records, batch_embeddings):
                embedded_records.append((record, embedding))
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
            continue
        except Exception:
            # Some OpenAI-compatible servers are happier with single inputs.
            pass

        for record in batch_records:
            try:
                embedding = _request_embeddings(
                    embedding_client=embedding_client,
                    model_name=model_name,
                    profile_name=profile_name,
                    chunk_records=[record],
                )[0]
                embedded_records.append((record, embedding))
            except Exception as exc:
                if warning_callback:
                    warning_callback(
                        f"Embedding failed: {record.source}, "
                        f"chunk {record.chunk_id} -> {exc}"
                    )
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return embedded_records
