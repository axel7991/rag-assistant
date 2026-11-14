from chunking import chunk_text
import pytest


def test_chunk_text():
    text = "one two three four five six seven"

    chunks = chunk_text(text, chunk_size=3, overlap=0)

    assert chunks == [
        "one two three",
        "four five six",
        "seven"
    ]




def test_chunk_size_zero():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0)



def test_chunk_text_with_overlap():
    text = "one two three four five six seven eight"

    chunks = chunk_text(
        text,
        chunk_size=4,
        overlap=2
    )

    assert chunks == [
        "one two three four",
        "three four five six",
        "five six seven eight"
    ]



def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three four five",
            chunk_size=4,
            overlap=4
        )



def test_default_overlap_with_small_chunk():
    text = "one two three four five"

    chunks = chunk_text(text, chunk_size=3)

    assert chunks == [
        "one two three",
        "four five"
    ]


def test_no_redundant_trailing_chunk():
    text = "one two three four five six seven eight"

    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert len(chunks) == 3
    assert chunks[-1] == "five six seven eight"