def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int | None = None
) -> list[str]:
    """Split text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap is None:
        overlap = min(50, chunk_size // 10)

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
        if i == 0 or i + overlap < len(words)
    ]