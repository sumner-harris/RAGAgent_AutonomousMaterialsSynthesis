from RAG.openai_compat import connection_is_configured, make_openai_client


def _build_prompt(query: str, context: str) -> str:
    return f"""
User query: {query}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
""".strip()


def generate_answer(
    *,
    query: str,
    context_text: str,
    model: str,
    system: str,
    api_key: str | None = None,
    base_url: str | None = None,
    enable_web_search: bool = False,
):
    if not connection_is_configured(api_key, base_url):
        raise ValueError("A model endpoint must be configured for generation.")

    client = make_openai_client(api_key=api_key, base_url=base_url)
    prompt = _build_prompt(query, context_text)

    kwargs = {"model": model, "instructions": system, "input": prompt}
    if enable_web_search:
        kwargs["tools"] = [{"type": "web_search_preview"}]

    response = client.responses.create(**kwargs)
    return response.output_text
