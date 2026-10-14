import os
import sys
from pathlib import Path
import chunking
import embedding


def ingest_file(pdf_path: str) -> bool:
    """
    Parses, chunks, and persists a PDF file's embeddings into ChromaDB.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        bool: True if ingestion was successful, False otherwise.
    """
    clean_path_str = pdf_path.strip().strip('"').strip("'")
    path = Path(clean_path_str)

    if not path.is_file():
        print(f"\n[Error] File not found at: '{clean_path_str}'. Please verify the path.")
        return False

    if path.suffix.lower() != ".pdf":
        print(f"\n[Error] '{path.name}' is not a PDF file. Please provide a valid .pdf file.")
        return False

    print(f"\n[1/3] Parsing & chunking PDF: '{path.name}'...")
    try:
        chunks = chunking.parse_and_chunk_pdf(str(path))
    except Exception as e:
        print(f"[Error] Failed to parse and chunk PDF: {e}")
        return False

    if not chunks:
        print("[Warning] No text content found in the PDF.")
        return False

    total_chunks = len(chunks)
    print(f"[2/3] Extracted {total_chunks} chunks.")
    print(f"[3/3] Generating embeddings and saving into ChromaDB...")

    for i, chunk in enumerate(chunks, start=1):
        embedding.add_document(
            doc_id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata
        )
        if i % 5 == 0 or i == total_chunks:
            print(f"  -> Progress: {i}/{total_chunks} chunks saved.")

    total_docs = embedding.collection.count()
    print("\n" + "=" * 50)
    print("           INGESTION SUCCESSFUL")
    print("=" * 50)
    print(f"- File ingested: {path.name}")
    print(f"- Chunks added:  {total_chunks}")
    print(f"- Total in DB:   {total_docs} records")
    print("=" * 50 + "\n")
    return True


def main():
    print("=" * 50)
    print("       DilliGuide - PDF Ingestion Pipeline")
    print("=" * 50)
    print(f"Connected to ChromaDB Collection: '{embedding.collection.name}'")
    print(f"Current documents in database:    {embedding.collection.count()}\n")

    # Check CLI arguments for clear flag
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("--clear", "-c", "--reset"):
            confirm = input("Are you sure you want to clear the entire database? (y/N): ").strip().lower()
            if confirm in ("y", "yes"):
                embedding.clear_database()
            else:
                print("Clear operation cancelled.")
            return
        pdf_path = arg
    else:
        try:
            pdf_path = input("Enter PDF file path (or type 'clear' to reset database): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

    if not pdf_path:
        print("No input entered. Exiting.")
        return

    # Check if user typed 'clear' or 'reset'
    if pdf_path.lower() in ("clear", "reset", "--clear", "-c"):
        confirm = input("Are you sure you want to clear the entire database? (y/N): ").strip().lower()
        if confirm in ("y", "yes"):
            embedding.clear_database()
        else:
            print("Clear operation cancelled.")
        return

    ingest_file(pdf_path)


if __name__ == "__main__":
    main()

