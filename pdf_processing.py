from pypdf import PdfReader
from chunking import chunk_text

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extract text and page numbers from a PDF."""

    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def extract_pdf_chunks(file_path: str) -> list[dict]:
    """Extract and chunk PDF text while preserving page numbers."""

    pages = extract_pdf_pages(file_path)
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk_index, text in enumerate(page_chunks):
            chunks.append({
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": text
            })

    return chunks