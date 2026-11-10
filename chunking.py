def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks containing at most chunk_size words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    words = text.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]