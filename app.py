def call_gpt(api_key: str, model: str, system: str, user: str) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (getattr(resp, "output_text", "") or "").strip()


def translate_text(api_key: str, model: str, text: str, target_lang: str) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    client = OpenAI(api_key=api_key)

    sys = (
        "You are a translation engine. Translate ONLY. "
        "Do not rewrite, summarize, add, remove, or change meaning. "
        "Preserve structure, headings like TITLE/BODY, numbers, tickers, and punctuation. "
        "Preserve paragraph breaks exactly."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": sys},
