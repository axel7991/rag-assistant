from chunking import chunk_text
import pytest


def test_chunk_text():
    text = "one two three four five six seven"

    chunks = chunk_text(text, chunk_size=3)

    assert chunks == [
        "one two three",
        "four five six",
        "seven"
    ]




def test_chunk_size_zero():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0)