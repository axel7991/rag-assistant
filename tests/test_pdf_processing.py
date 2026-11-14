from unittest.mock import patch, MagicMock
from pdf_processing import extract_pdf_pages
from pdf_processing import extract_pdf_pages, extract_pdf_chunks


def test_extract_pdf_pages():
    page1 = MagicMock()
    page1.extract_text.return_value = "Introduction to AI"

    page2 = MagicMock()
    page2.extract_text.return_value = ""

    page3 = MagicMock()
    page3.extract_text.return_value = "Machine learning methods"

    with patch("pdf_processing.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [page1, page2, page3]

        result = extract_pdf_pages("sample.pdf")

    assert result == [
        {"page_number": 1, "text": "Introduction to AI"},
        {"page_number": 3, "text": "Machine learning methods"}
    ]



def test_extract_pdf_chunks():
    fake_pages = [
        {"page_number": 1, "text": "one two three four five"},
        {"page_number": 3, "text": "six seven eight nine ten"}
    ]

    with patch("pdf_processing.extract_pdf_pages", return_value=fake_pages):
        with patch(
            "pdf_processing.chunk_text",
            side_effect=lambda text: [text[:10], text[10:]]
        ):
            result = extract_pdf_chunks("sample.pdf")

    assert len(result) == 4

    assert result[0]["page_number"] == 1
    assert result[0]["chunk_index"] == 0

    assert result[1]["page_number"] == 1
    assert result[1]["chunk_index"] == 1

    assert result[2]["page_number"] == 3
    assert result[2]["chunk_index"] == 0

    assert result[3]["page_number"] == 3
    assert result[3]["chunk_index"] == 1