import os
import sys
from collections import Counter
from pathlib import Path
import chunking
import embedding
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()


def ingest_single_file(file_path: Path, batch_size: int = 20) -> tuple[int, Counter]:
    """
    Parses, chunks, and persists a single PDF or Markdown file's embeddings into ChromaDB.

    Args:
        file_path (Path): Path object to the .pdf or .md file.
        batch_size (int): Number of chunks to embed and upsert per batch.

    Returns:
        tuple[int, Counter]: (total_chunks_added, category_counts)
    """
    suffix = file_path.suffix.lower()
    if suffix not in (".pdf", ".md"):
        console.print(f"[bold red][Error][/bold red] '{file_path.name}' is not a supported file (.pdf or .md).")
        return 0, Counter()

    console.print(f"\n[cyan][1/3][/cyan] Parsing & chunking {suffix.upper()} file: [bold yellow]'{file_path.name}'[/bold yellow]...")
    try:
        chunks = chunking.parse_and_chunk_pdf(str(file_path))
    except Exception as e:
        console.print(f"[bold red][Error][/bold red] Failed to parse and chunk file: {e}")
        return 0, Counter()

    if not chunks:
        console.print(f"[bold yellow][Warning][/bold yellow] No text content found in '{file_path.name}'.")
        return 0, Counter()

    total_chunks = len(chunks)
    console.print(f"[cyan][2/3][/cyan] Extracted [bold green]{total_chunks}[/bold green] chunks with semantic hierarchy & context.")
    console.print(f"[cyan][3/3][/cyan] Generating embeddings in batches of {batch_size} and saving to ChromaDB...")

    # Batched loop to minimize network roundtrips
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_ids = [c.id for c in batch]
        batch_texts = [c.text for c in batch]
        batch_metadatas = [c.metadata for c in batch]

        embedding.add_documents(
            doc_ids=batch_ids,
            texts=batch_texts,
            metadatas=batch_metadatas
        )

        processed = min(i + len(batch), total_chunks)
        console.print(f"  [dim]-> Progress: {processed}/{total_chunks} chunks saved.[/dim]")

    category_counts = Counter(chunk.metadata.get("category", "general") for chunk in chunks)
    return total_chunks, category_counts


def ingest_file(file_path: str, batch_size: int = 20) -> bool:
    """
    Parses, chunks, and persists embeddings into ChromaDB using batching.
    Accepts a single file (.pdf or .md) OR a directory/folder containing .pdf and .md files.

    Args:
        file_path (str): The file or folder path.
        batch_size (int): Number of chunks to embed and upsert per batch (default: 20).

    Returns:
        bool: True if ingestion succeeded for at least one file, False otherwise.
    """
    clean_path_str = file_path.strip().strip('"').strip("'")
    path = Path(clean_path_str)

    if not path.exists():
        console.print(f"\n[bold red][Error][/bold red] Path not found at: '{clean_path_str}'. Please verify the path.")
        return False

    # Handle directory ingestion
    if path.is_dir():
        supported_files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".md")]
        )
        if not supported_files:
            console.print(f"[bold yellow][Warning][/bold yellow] No .pdf or .md files found in directory: '{path.resolve()}'")
            return False

        console.print(
            Panel.fit(
                f"[bold cyan]Directory Ingestion Mode[/bold cyan]\n"
                f"Found [bold green]{len(supported_files)}[/bold green] documents to ingest in: [dim]{path.resolve()}[/dim]",
                border_style="cyan"
            )
        )

        total_chunks_all = 0
        overall_categories = Counter()
        files_ingested = 0

        for f in supported_files:
            chunks_count, cat_counts = ingest_single_file(f, batch_size=batch_size)
            if chunks_count > 0:
                files_ingested += 1
                total_chunks_all += chunks_count
                overall_categories.update(cat_counts)

        # Print overall summary
        table = Table(title="Directory Ingestion Summary", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Chunks Ingested", justify="right", style="green")
        for cat, count in sorted(overall_categories.items()):
            table.add_row(cat, str(count))

        console.print("\n")
        console.print(table)
        console.print(
            Panel.fit(
                f"[bold green]Ingestion Complete![/bold green]\n"
                f"• Files Ingested: [bold]{files_ingested}/{len(supported_files)}[/bold]\n"
                f"• Total Chunks Added: [bold]{total_chunks_all}[/bold]\n"
                f"• Total in ChromaDB: [bold yellow]{embedding.collection.count()}[/bold yellow] records",
                border_style="green"
            )
        )
        return files_ingested > 0

    # Handle single file (.pdf or .md)
    if path.suffix.lower() not in (".pdf", ".md"):
        console.print(f"\n[bold red][Error][/bold red] '{path.name}' is not a supported file (.pdf or .md).")
        return False

    total_chunks, category_counts = ingest_single_file(path, batch_size=batch_size)
    if total_chunks == 0:
        return False

    total_docs = embedding.collection.count()

    table = Table(title=f"Category Breakdown: {path.name}", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Chunk Count", justify="right", style="green")
    for cat, count in sorted(category_counts.items()):
        table.add_row(cat, str(count))

    console.print("\n")
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]Ingestion Successful![/bold green]\n"
            f"• File Ingested: [bold]{path.name}[/bold]\n"
            f"• Chunks Added:  [bold]{total_chunks}[/bold]\n"
            f"• Total in DB:   [bold yellow]{total_docs}[/bold yellow] records",
            border_style="green"
        )
    )
    return True


def reset_and_reingest(target_pdf: str = "lalquila.pdf"):
    """
    One-time cleanup: Cleans orphan test collections, resets dilli_guide_collection,
    and re-ingests target PDF to start 100% clean with v2 chunks.
    """
    console.print(Panel("[bold yellow]Database Reset & Clean Ingestion[/bold yellow]\nCleaning test collections and resetting ChromaDB...", border_style="yellow"))
    embedding.clear_database()

    pdf_path = Path(target_pdf)
    if pdf_path.exists():
        console.print(f"[bold cyan]Re-ingesting benchmark corpus:[/bold cyan] [green]{pdf_path.name}[/green]")
        ingest_file(str(pdf_path))
    else:
        console.print(f"[dim]No default '{target_pdf}' found in workspace. Database is clean and ready.[/dim]")


def main():
    console.print("=" * 60)
    console.print("         [bold yellow]DilliGuide - Ingestion & Knowledge Engine[/bold yellow]")
    console.print("=" * 60)
    console.print(f"Connected to ChromaDB Collection: [bold green]'{embedding.collection.name}'[/bold green]")
    console.print(f"Current documents in database:    [bold yellow]{embedding.collection.count()}[/bold yellow]\n")

    # Check CLI arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("--clear", "-c", "--reset"):
            confirm = input("Are you sure you want to clear the entire database? (y/N): ").strip().lower()
            if confirm in ("y", "yes"):
                embedding.clear_database()
            else:
                console.print("[dim]Clear operation cancelled.[/dim]")
            return
        elif arg == "--reingest":
            reset_and_reingest("lalquila.pdf")
            return
        target_path = arg
    else:
        try:
            target_path = input("Enter file (.pdf/.md) or directory path (or type 'clear'/'reingest'): ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Operation cancelled.[/dim]")
            return

    if not target_path:
        console.print("[dim]No input entered. Exiting.[/dim]")
        return

    # Check if user typed 'clear' or 'reset' or 'reingest'
    if target_path.lower() in ("clear", "reset", "--clear", "-c"):
        confirm = input("Are you sure you want to clear the entire database? (y/N): ").strip().lower()
        if confirm in ("y", "yes"):
            embedding.clear_database()
        else:
            console.print("[dim]Clear operation cancelled.[/dim]")
        return
    elif target_path.lower() in ("reingest", "--reingest"):
        reset_and_reingest("lalquila.pdf")
        return

    ingest_file(target_path)


if __name__ == "__main__":
    main()
