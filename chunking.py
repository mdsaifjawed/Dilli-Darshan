import os
from pathlib import Path
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter


class PDFChunk(dict):
    """
    Represents a text chunk extracted from a PDF.
    
    Supports both dictionary-style access (`chunk["text"]`, `chunk["metadata"]`, `chunk["id"]`)
    and attribute-style access (`chunk.text`, `chunk.metadata`, `chunk.id`, `chunk.page_content`).
    """
    def __init__(self, id: str, text: str, metadata: dict):
        super().__init__(id=id, text=text, metadata=metadata)
        self.id = id
        self.text = text
        self.metadata = metadata
        self.page_content = text  # LangChain compatibility

    def __repr__(self):
        return f"PDFChunk(id='{self.id}', text_len={len(self.text)}, metadata={self.metadata})"


def parse_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    by_page: bool = True
) -> list[PDFChunk]:
    """
    Parses a PDF into Markdown using pymupdf4llm and splits it into structured
    chunks using langchain_text_splitters.MarkdownTextSplitter.

    Args:
        pdf_path (str): Path to the target PDF file.
        chunk_size (int): Maximum character length per chunk (default: 1000).
        chunk_overlap (int): Number of overlapping characters between chunks (default: 150).
        by_page (bool): If True, preserves exact page numbers in metadata per page (default: True).

    Returns:
        list[PDFChunk]: A list of chunk objects, each containing:
            - 'id': Unique ID for vector DB upsertion (e.g. 'docname_p1_c0')
            - 'text': Chunk markdown text
            - 'metadata': Metadata dict with source filename, page number, chunk index, etc.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # Clean filename stem for safe database IDs
    clean_stem = path.stem.replace(" ", "_").lower()

    # Markdown-aware splitter preserves tables, headings, and lists
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks: list[PDFChunk] = []

    if by_page:
        # Extract page by page to track precise page numbers
        pages_data = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        total_pages = len(pages_data)

        for page_info in pages_data:
            page_meta = page_info.get("metadata", {})
            page_num = page_meta.get("page_number", 1)
            raw_text = page_info.get("text", "")

            if not raw_text or not raw_text.strip():
                continue

            split_texts = splitter.split_text(raw_text)

            for chunk_idx, chunk_text in enumerate(split_texts):
                chunk_id = f"{clean_stem}_p{page_num}_c{chunk_idx}"
                metadata = {
                    "source": path.name,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "total_pages": total_pages
                }
                chunks.append(PDFChunk(id=chunk_id, text=chunk_text.strip(), metadata=metadata))
    else:
        # Extract entire PDF as continuous Markdown
        full_markdown = pymupdf4llm.to_markdown(str(path))
        if full_markdown and full_markdown.strip():
            split_texts = splitter.split_text(full_markdown)
            for chunk_idx, chunk_text in enumerate(split_texts):
                chunk_id = f"{clean_stem}_c{chunk_idx}"
                metadata = {
                    "source": path.name,
                    "chunk_index": chunk_idx
                }
                chunks.append(PDFChunk(id=chunk_id, text=chunk_text.strip(), metadata=metadata))

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
        print(f"Parsing and chunking: {target_pdf}")
        parsed_chunks = parse_and_chunk_pdf(target_pdf)
        print(f"Total chunks created: {len(parsed_chunks)}")
        if parsed_chunks:
            print("\nFirst chunk preview:")
            print(f"- ID: {parsed_chunks[0].id}")
            print(f"- Metadata: {parsed_chunks[0].metadata}")
            print(f"- Text:\n{parsed_chunks[0].text[:300]}...")
    else:
        print("Usage: python chunking.py <path_to_pdf>")
