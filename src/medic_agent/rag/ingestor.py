import io

from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _chunk_text(text: str, filename: str) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {"text": chunk_text, "source_filename": filename, "chunk_index": index}
            )
            index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )
    return _chunk_text(full_text.strip(), filename)


def load_text(text: str, filename: str) -> list[dict]:
    return _chunk_text(text.strip(), filename)
