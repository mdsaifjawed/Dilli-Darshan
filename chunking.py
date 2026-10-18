import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class PDFChunk(dict):
    """
    Represents a text chunk extracted from a PDF or Markdown document.
    
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


def clean_header_text(text: str | None) -> str:
    """Removes markdown formatting (*, _, #) and normalizes whitespace.
    
    If the cleaned header string exceeds 80 characters (e.g. when PDF extraction
    merges a heading and its first sentence), splits at the first sentence/colon or
    truncates cleanly at 80 characters to prevent paragraph duplication in context headers.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[*_#]", "", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) > 80:
        for delimiter in [". ", ": ", ".\n", ":\n", " - "]:
            if delimiter in cleaned:
                cleaned = cleaned.split(delimiter, 1)[0].strip()
                break
        if len(cleaned) > 80:
            if " " in cleaned[:80]:
                cleaned = cleaned[:80].rsplit(" ", 1)[0].strip()
            else:
                cleaned = cleaned[:80].strip()

    return cleaned.rstrip(",;:- ")


def detect_category(section_title: str) -> str:
    """
    Inspects combined section and subsection title (case-insensitive) to assign
    semantic category tags for metadata filtering based on strict priority order.
    """
    title_lower = (section_title or "").lower()

    # 1. Quick Facts (Highest priority so summaries/reference data are categorized properly)
    if any(k in title_lower for k in [
        "quick reference", "summary", "entity retrieval", "quick facts",
        "introduction", "overview", "final summary", "important tips",
        "worth visiting", "suggested visit plan", "one-day visit", "tips",
        "itinerary", "highlights", "facts"
    ]):
        return "quick_facts"

    # 2. Timings and Tickets (Takes precedence over general terms like structure)
    if any(k in title_lower for k in [
        "timing", "ticket", "guideline", "show", "security", "closure", "fee", "operating",
        "opening days", "visiting timings", "entry ticket", "other costs", "other expenses",
        "dress code", "etiquette", "visitor rules", "budget", "entry", "cost", "price",
        "hours", "pass", "admission", "fare"
    ]):
        return "timings_and_tickets"

    # 3. Transit and Metro
    if any(k in title_lower for k in [
        "reach", "transit", "metro", "rail", "bus", "airport", "station",
        "how to reach", "railway station", "by car", "by delhi metro", "local transport",
        "route", "direction", "dmrc", "transport"
    ]):
        return "transit_and_metro"

    # 4. Food and Dining
    if any(k in title_lower for k in [
        "restaurant", "culinary", "dining", "food", "eater", "cafe",
        "cuisine", "snack", "street food", "delicac",
        "food court", "haldiram", "lunch", "coffee", "cafeteria",
        "chaat", "sweet", "mithai", "eateries", "dish", "bakery", "biryani"
    ]):
        return "food_and_dining"

    # 5. History and Architecture ("structure" removed to avoid misclassifying fee structure)
    if any(k in title_lower for k in [
        "history", "heritage", "architect", "chronolog", "mughal", "dynasty", "fortress",
        "significance", "special features", "main attractions", "locomotives", "gallery", "exhibits",
        "attraction", "gate", "palace", "mahal", "monument", "masjid", "fort", "diwan", "baoli", "tomb"
    ]):
        return "history_and_architecture"

    return "general"


def chunk_markdown_text(
    full_markdown: str,
    source_name: str,
    place_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> list[PDFChunk]:
    """
    Core Markdown splitting and chunking logic.
    Splits semantically on Markdown headers, sub-splits large sections, injects context headers,
    and categorizes chunks.

    Args:
        full_markdown (str): Continuous markdown text string.
        source_name (str): Filename or source identifier (e.g. 'lalquila.pdf' or 'sunder_nursery.md').
        place_id (str): Canonical identifier for the monument/place.
        chunk_size (int): Maximum character length per chunk (default: 1200).
        chunk_overlap (int): Number of overlapping characters between chunks (default: 150).

    Returns:
        list[PDFChunk]: A list of chunk objects with 'id', 'text', and 'metadata'.
    """
    if not full_markdown or not full_markdown.strip():
        return []

    clean_stem = place_id.replace(" ", "_").lower()

    # 1. Derive canonical monument_title from place_id (stripped of suffixes like _guide)
    CANONICAL_PLACE_TITLES = {
        "air_force_museum": "Air Force Museum",
        "akshardham": "Akshardham Temple",
        "akshardham_guide": "Akshardham Temple",
        "craft_museum": "National Crafts Museum",
        "gandhi_samriti_museum": "Gandhi Smriti Museum",
        "india_gate": "India Gate",
        "indra_gandhi_museum": "Indira Gandhi Memorial Museum",
        "jama_masjid": "Jama Masjid",
        "jama_masjid_guide": "Jama Masjid",
        "jantar_mantar": "Jantar Mantar",
        "lal_qila": "Lal Qila",
        "lalquila": "Lal Qila",
        "pm_snagyaly": "Pradhanmantri Sangrahalaya",
        "qutub_minar": "Qutub Minar",
        "rail_museum": "National Rail Museum",
        "red_fort": "Lal Qila",
        "lotus_temple": "Lotus Temple",
        "humayuns_tomb": "Humayun's Tomb",
        "sunder_nursery": "Sunder Nursery"
    }

    raw_base = clean_stem
    for sfx in ("_guide", "-guide", "_pdf"):
        if raw_base.endswith(sfx):
            raw_base = raw_base[:-len(sfx)]

    canonical_monument_title = (
        CANONICAL_PLACE_TITLES.get(clean_stem)
        or CANONICAL_PLACE_TITLES.get(raw_base)
        or raw_base.replace("_", " ").strip().title()
    )

    # 1. Semantic splitting on Markdown headers
    headers_to_split_on = [
        ("#", "monument"),
        ("##", "section"),
        ("###", "subsection")
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    header_docs = header_splitter.split_text(full_markdown)

    # Fallback if no markdown headers were found in document
    if not header_docs:
        try:
            from langchain_core.documents import Document
            header_docs = [Document(page_content=full_markdown, metadata={})]
        except Exception:
            class _SimpleDoc:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata
            header_docs = [_SimpleDoc(page_content=full_markdown, metadata={})]

    # 2. Recursive splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks: list[PDFChunk] = []
    chunk_counter = 0

    def looks_like_section_header(hdr: str | None) -> bool:
        if not hdr:
            return False
        # Strip markdown formatting (*, _, #) first
        h = clean_header_text(hdr).strip()
        if not h:
            return False
        # Starts with number/letter like "1.", "2.", "A.", "B.", "Morning", "Evening", "From ", "By "
        if re.match(r"^(\d+[\.\)]|[a-zA-Z][\.\)]|\b(morning|evening|afternoon|first\s|from\s|by\s))", h, re.IGNORECASE):
            return True
        # Matches any category keyword in detect_category
        if detect_category(h) != "general":
            return True
        # Check if it is not the document's main title
        clean_canon = canonical_monument_title.lower()
        if clean_canon and h.lower() != clean_canon and h.lower() != raw_base.replace("_", " "):
            if any(term in h.lower() for term in ("tourist guide", "visitor guide")):
                return False
            # Any subheader that isn't the document's main monument title
            return True
        return False

    for doc in header_docs:
        raw_monument = doc.metadata.get("monument")
        raw_section = doc.metadata.get("section")
        raw_subsection = doc.metadata.get("subsection")

        cleaned_monument = clean_header_text(raw_monument)
        cleaned_section = clean_header_text(raw_section)
        cleaned_subsection = clean_header_text(raw_subsection)

        # Fix H1 (#) Header Bug:
        # If raw_section is None or empty, AND raw_monument looks like a section header:
        if not cleaned_section or cleaned_section == "General":
            if looks_like_section_header(raw_monument):
                section = cleaned_monument
                monument = canonical_monument_title
            else:
                if cleaned_monument and cleaned_monument.lower() in ("delhi tourist guide", "visitor guide", "tourist guide"):
                    monument = canonical_monument_title
                else:
                    monument = cleaned_monument or canonical_monument_title
                section = cleaned_section or "General"
        else:
            monument = canonical_monument_title if (cleaned_monument and cleaned_monument.lower() in ("delhi tourist guide", "visitor guide")) else (cleaned_monument or canonical_monument_title)
            section = cleaned_section

        subsection = cleaned_subsection or ""

        # Always run detect_category() on f"{raw_monument or ''} {section} {subsection}"
        category_search_text = f"{raw_monument or ''} {section} {subsection}".strip()
        category = detect_category(category_search_text)

        # Context header to prevent orphaned chunks
        if subsection:
            context_header = f"[Place: {monument} | Section: {section} > {subsection}]\n"
        else:
            context_header = f"[Place: {monument} | Section: {section}]\n"

        # Sub-split long section content
        sub_chunks = text_splitter.split_text(doc.page_content)

        for sub_chunk in sub_chunks:
            sub_chunk_stripped = sub_chunk.strip()
            if not sub_chunk_stripped:
                continue

            # Injected context text
            chunk_text = f"{context_header}{sub_chunk_stripped}"
            chunk_id = f"{clean_stem}_c{chunk_counter}"

            metadata = {
                "source": source_name,
                "place_id": clean_stem,
                "monument": monument,
                "section": section,
                "subsection": subsection,  # Guaranteed "" if None
                "category": category,
                "chunk_index": chunk_counter
            }

            chunks.append(PDFChunk(id=chunk_id, text=chunk_text, metadata=metadata))
            chunk_counter += 1

    return chunks


def parse_and_chunk_pdf(
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> list[PDFChunk]:
    """
    Parses a PDF or Markdown file, extracts Markdown, and delegates to chunk_markdown_text.

    Args:
        file_path (str): Path to the target PDF or MD file.
        chunk_size (int): Maximum character length per chunk (default: 1200).
        chunk_overlap (int): Number of overlapping characters between chunks (default: 150).

    Returns:
        list[PDFChunk]: A list of chunk objects with 'id', 'text', and 'metadata'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at: {file_path}")

    clean_stem = path.stem.replace(" ", "_").lower()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        full_markdown = pymupdf4llm.to_markdown(str(path))
    elif suffix == ".md":
        full_markdown = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats are .pdf and .md")

    return chunk_markdown_text(
        full_markdown=full_markdown,
        source_name=path.name,
        place_id=clean_stem,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# Alias for generalized file ingestion
parse_and_chunk_file = parse_and_chunk_pdf


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        console.print(f"[bold cyan]Parsing and chunking file:[/bold cyan] {target_file}")
        parsed_chunks = parse_and_chunk_pdf(target_file)
        console.print(f"[bold green]Total chunks created:[/bold green] {len(parsed_chunks)}")
        if parsed_chunks:
            first = parsed_chunks[0]
            console.print(
                Panel(
                    Markdown(first.text),
                    title=f"[bold yellow]Chunk Preview: {first.id}[/bold yellow]",
                    subtitle=f"[dim]{first.metadata}[/dim]",
                    border_style="cyan"
                )
            )
    else:
        console.print("[dim]Usage: python chunking.py <path_to_pdf_or_md>[/dim]")
