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


def ingest_autonomous_place(place_name: str) -> bool:
    """
    Autonomously discovers, fetches, structures, and ingests a comprehensive
    guide for any Delhi landmark/neighborhood using the TinyFish SDK and Gemini.
    Ensures all 5 sections are 100% complete without placeholders.
    """
    clean_name = place_name.strip().strip('"').strip("'")
    if not clean_name:
        console.print("[bold red][Error][/bold red] Place name cannot be empty.")
        return False

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]🌐 Autonomous Place Ingestion via TinyFish Engine[/bold cyan]\n"
            f"Target Place: [bold yellow]'{clean_name}'[/bold yellow]\n"
            f"[dim]Searching web, scraping live markdown, synthesizing 5-section guide, and embedding...[/dim]",
            border_style="cyan"
        )
    )

    import retrieval

    # 1. TinyFish Live Web Search & Fetch
    with console.status("[bold cyan]Searching and fetching live web markdown via TinyFish...", spinner="dots"):
        search_query = f"{clean_name} New Delhi history architecture timings entry ticket fee nearest metro station iconic food cafes"
        tf_data = retrieval.tinyfish_live_search_and_fetch(search_query, max_urls=3)

    if not tf_data.get("success") or not tf_data.get("markdown_context"):
        console.print(f"[bold red][Error][/bold red] TinyFish could not fetch web markdown for '{clean_name}': {tf_data.get('error', 'No content')}")
        return False

    console.print(f"[bold green]✓ Live web pages extracted ({len(tf_data['urls'])}):[/bold green]")
    for s in tf_data["sources"]:
        console.print(f"  • [blue underline]{s['title']}[/blue underline]: [dim]{s['uri']}[/dim]")

    # 2. Structure into 5-Section Markdown Guide using Gemini
    with console.status("[bold cyan]Synthesizing complete 5-section guidebook with Gemini...", spinner="dots"):
        structuring_prompt = f"""You are an expert travel editor for Dilli Darshan.
Create our standardized, high-quality 5-section Dilli Darshan guidebook for: {clean_name}.

Live Web Research Context:
{tf_data['markdown_context']}

CRITICAL QUALITY & FACTUALITY REQUIREMENTS:
1. EVERY SINGLE SECTION (1 through 5) MUST BE THOROUGHLY AND COMPLETELY FILLED with verified, factual travel information.
2. NEVER output placeholder text such as "*Information is not available in the provided context*", "N/A", or "Details not provided". Use the provided live web research context to detail the history, architecture, timings, tickets, metro route, eateries, and visitor tips.
3. Keep a strictly secular and professional tone (no religious salutations like "Namaste").

Output Structure:
# {clean_name}
## 1. History & Architecture
<In-depth historical context, founding era, architecture style, builder, materials, historical significance>

## 2. Timings & Tickets
<Visiting hours, weekly off day, entry fees for Indians and Foreigners, ticket booking info/portal, audio guide fees>

## 3. How to Reach & Metro
<Nearest Delhi Metro station, line color, nearest gate number, walking/auto distance, transit guidance>

## 4. Nearby Iconic Food Spots
<Famous nearby eateries, cafes, restaurants, street food specialties with recommended dishes and price range>

## 5. Quick Reference & Facts
<Key highlights, recommended duration, photography rules, visitor tips, accessibility>

Do not include backticks (```markdown ... ```) around the entire output. Return the raw Markdown starting with '# '."""

        try:
            response, _ = retrieval.execute_gemini_generation(structuring_prompt, enable_search=False)
            structured_md = (response.text or "").strip()
            if structured_md.startswith("```markdown"):
                structured_md = structured_md[len("```markdown"):].strip()
            if structured_md.startswith("```"):
                structured_md = structured_md[len("```"):].strip()
            if structured_md.endswith("```"):
                structured_md = structured_md[:-3].strip()
        except Exception as e:
            console.print(f"[bold red][Error][/bold red] Gemini synthesis failed: {e}")
            return False

    # 3. Save to data/<place_id>.md, chunk, and embed into ChromaDB
    with console.status("[bold cyan]Saving to data/ and embedding chunks into ChromaDB...", spinner="dots"):
        ok, count, file_path = retrieval.auto_cache_to_chromadb(clean_name, structured_md)

    if not ok:
        console.print(f"[bold red][Error][/bold red] Failed to chunk and persist guide for '{clean_name}'.")
        return False

    # 4. Display results & category breakdown
    target_p = Path(file_path)
    chunks = chunking.parse_and_chunk_pdf(str(target_p))
    cat_counts = Counter(c.metadata.get("category", "general") for c in chunks)

    table = Table(title=f"Category Breakdown: {target_p.name}", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Chunk Count", justify="right", style="green")
    for cat, c_count in sorted(cat_counts.items()):
        table.add_row(cat, str(c_count))

    console.print("\n")
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]Autonomous Ingestion Successful![/bold green]\n"
            f"• Place Ingested: [bold]{clean_name}[/bold]\n"
            f"• Guide Saved:    [yellow]{file_path}[/yellow]\n"
            f"• Chunks Added:   [bold]{count}[/bold]\n"
            f"• Total in DB:    [bold yellow]{embedding.collection.count()}[/bold yellow] records",
            border_style="green"
        )
    )
    return True


def reset_and_reingest(target_pdf: str = "lalquila.pdf"):
    """
    One-time cleanup: Cleans orphan test collections, resets dilli_darshan_collection,
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
    console.print("         [bold yellow]Dilli Darshan - Ingestion & Knowledge Engine[/bold yellow]")
    console.print("=" * 60)
    console.print(f"Connected to ChromaDB Collection: [bold green]'{embedding.collection.name}'[/bold green]")
    console.print(f"Current documents in database:    [bold yellow]{embedding.collection.count()}[/bold yellow]")
    backup_info = f" (Backups: [green]{embedding.client.backup_label}[/green])" if getattr(embedding.client, "backup_label", None) else ""
    console.print(f"Gemini API Key:                   [bold green]{getattr(embedding.client, 'active_label', 'API_Key')}[/bold green]{backup_info}\n")

    # Check CLI arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("--tinyfish", "-t"):
            place_parts = sys.argv[2:]
            if place_parts:
                place = " ".join(place_parts).strip()
                ingest_autonomous_place(place)
            else:
                console.print("[bold red]Please specify a place name, e.g.:[/bold red] python ingesting.py --tinyfish \"Lotus Temple\"")
            return
        elif arg in ("--clear", "-c", "--reset"):
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
            target_path = input("Enter file (.pdf/.md), directory path, or 'tinyfish <Place>' (or 'clear'/'reingest'): ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Operation cancelled.[/dim]")
            return

    if not target_path:
        console.print("[dim]No input entered. Exiting.[/dim]")
        return

    # Check if user typed 'tinyfish <Place>'
    if target_path.lower().startswith(("tinyfish ", "tf ", "--tinyfish ")):
        parts = target_path.split(" ", 1)
        if len(parts) > 1 and parts[1].strip():
            ingest_autonomous_place(parts[1].strip())
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
