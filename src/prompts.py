from src.models import RetrievalResult


FALLBACK = "Insufficient information in the provided document."


def build_prompt(question: str, sources: list[RetrievalResult]) -> str:
    context = []
    for source in sources:
        pages = ", ".join(str(page) for page in source.chunk.page_numbers)
        context.append(
            f"[Source {source.rank}] Document: {source.chunk.document}\n"
            f"Pages: {pages}\n"
            f"Content: {source.chunk.text}"
        )
    context_text = "\n\n".join(context) if context else "(No retrieved context)"
    return (
        "You are a precise financial-report research assistant.\n"
        "Use only the retrieved document context below. Never invent financial values, "
        "periods, currencies, units, percentages, or forecasts.\n"
        "Distinguish reported facts from interpretations. Preserve exact reporting periods "
        "and accounting units. Cite claims using [Source 1], [Source 2], etc.\n"
        f"If the context is insufficient, return exactly: {FALLBACK}\n"
        "A citation is valid only when the cited source exists and supports the claim.\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
