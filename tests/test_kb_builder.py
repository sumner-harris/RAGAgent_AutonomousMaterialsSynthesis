from RAG.retrieval.kb_builder import _write_graphrag_input_files


def test_write_graphrag_input_files_preserves_unicode_with_utf8(tmp_path):
    input_dir = tmp_path / "input"
    text = "The notation includes psi \u03c8 and italic a \U0001d44e."

    _write_graphrag_input_files(
        texts={"Harris et al ACS Nano 17 2472 2023 SI.pdf": text},
        input_dir=input_dir,
    )

    written = (input_dir / "Harris et al ACS Nano 17 2472 2023 SI.txt").read_text(
        encoding="utf-8"
    )
    assert written == text
