import os
import re
import sys
import difflib
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure UTF-8 output on Windows consoles to support rich emojis and formatting
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

import embedding
import chunking
import memory
from google.genai import types

console = Console()

MODEL_ID = "gemini-3.8-flash"
FALLBACK_MODEL_ID = "gemini-3.6-flash"
COUNTERMEASURE_MODEL_ID = "gemini-2.5-flash"

# Keyword dictionary for zero-latency category routing
CATEGORY_KEYWORDS = {
    "history_and_architecture": [
        "history", "heritage", "architect", "chronolog", "structure",
        "built", "emperor", "dynasty", "mughal", "historical", "monument",
        "shah jahan", "complex", "origin", "story", "significance", "special features",
        "main attractions", "locomotives", "gallery", "exhibits", "gate", "palace",
        "mahal", "masjid", "fort", "diwan", "baoli", "tomb"
    ],
    "timings_and_tickets": [
        "timing", "ticket", "guideline", "show", "security", "closure",
        "fee", "operating", "open", "close", "cost", "price", "entry", "hours",
        "schedule", "entry fee", "pass", "booking", "opening days", "visiting timings",
        "entry ticket", "other costs", "other expenses", "dress code", "etiquette",
        "visitor rules", "budget", "admission", "fare"
    ],
    "transit_and_metro": [
        "reach", "transit", "metro", "rail", "bus", "airport", "station",
        "direction", "route", "how to get", "nearest", "gate", "line", "dmrc",
        "distance", "cab", "auto", "how to reach", "railway station", "by car",
        "by delhi metro", "local transport", "transport"
    ],
    "food_and_dining": [
        "restaurant", "culinary", "dining", "food", "eater", "cafe",
        "cuisine", "snack", "street food", "delicac", "eat", "lunch", "dinner",
        "breakfast", "chaat", "mithai", "biryani", "sweet", "famous food",
        "food court", "haldiram", "coffee", "cafeteria", "eateries", "dish", "bakery"
    ],
    "quick_facts": [
        "quick reference", "summary", "entity retrieval", "fact", "overview", "info", "highlights",
        "introduction", "final summary", "important tips", "worth visiting", "suggested visit plan",
        "one-day visit", "tips", "itinerary", "facts"
    ],
}

# Real-time / Live info triggers
LIVE_KEYWORDS = [
    "today", "now", "current", "live", "2026", "event", "weather",
    "crowd", "traffic", "currently", "this week", "festival", "exhibition",
    "celebration", "open today", "closed today"
]

# Known Delhi tourist landmarks and monuments for place mismatch detection
KNOWN_DELHI_PLACES = {
    "red_fort": ["red fort", "lal quila", "lal qila", "lalquila", "lahori gate", "meena bazaar", "diwan-i-am"],
    "lal_qila": ["lal qila", "lal quila", "red fort", "lalquila", "lahori gate", "meena bazaar", "diwan-i-am"],
    "sunder_nursery": ["sunder nursery", "sundar nursery", "heritage park"],
    "humayuns_tomb": ["humayun tomb", "humayun's tomb", "humayuns tomb"],
    "humayun's_tomb": ["humayun's tomb", "humayun tomb", "humayuns tomb"],
    "cannaught_place": ["cannaught place", "connaught place", "cp", "rajiv chowk", "inner circle", "outer circle"],
    "connaught_place": ["connaught place", "cannaught place", "cp", "rajiv chowk", "inner circle", "outer circle"],
    "qutub_minar": ["qutub minar", "qutb minar", "mehrauli", "iron pillar"],
    "india_gate": ["india gate", "kartavya path", "amar jawan jyoti", "war memorial"],
    "lotus_temple": ["lotus temple", "bahai house of worship", "bahai temple"],
    "akshardham": ["akshardham", "swaminarayan akshardham", "akshardham temple", "akshardham guide"],
    "akshardham_guide": ["akshardham", "swaminarayan akshardham", "akshardham temple", "akshardham guide"],
    "jama_masjid": ["jama masjid", "old delhi mosque", "jama masjid guide"],
    "jama_masjid_guide": ["jama masjid", "old delhi mosque", "jama masjid guide"],
    "lodhi_garden": ["lodhi garden", "lodi garden", "sikandar lodi"],
    "chandni_chowk": ["chandni chowk", "paranthe wali gali", "khari baoli"],
    "bangla_sahib": ["bangla sahib", "gurudwara bangla sahib"],
    "hauz_khas": ["hauz khas", "hauz khas village", "deer park"],
    "jantar_mantar": ["jantar mantar", "samrat yantra"],
    "purana_qila": ["purana qila", "old fort", "qila-i-kuhna"],
    "safdarjung_tomb": ["safdarjung tomb", "safdarjang tomb", "safdarjung"],
    "national_museum": ["national museum", "janpath museum"],
    "raj_ghat": ["raj ghat", "rajghat", "gandhi memorial"],
    "agrasen_ki_baoli": ["agrasen ki baoli", "ugrasen ki baoli", "baoli"],
    "rashtrapati_bhavan": ["rashtrapati bhavan", "amrit udyan", "mughal gardens"],
    "air_force_museum": ["air force museum", "indian air force museum", "palam museum"],
    "craft_museum": ["craft museum", "crafts museum", "national crafts museum", "hastkala academy"],
    "gandhi_samriti_museum": ["gandhi smriti", "gandhi samriti", "birla house", "old birla house"],
    "indra_gandhi_museum": ["indira gandhi memorial", "indira gandhi museum", "indra gandhi museum"],
    "pm_snagyaly": ["pradhanmantri sangrahalaya", "pm sangrahalaya", "pm museum", "teen murti"],
    "rail_museum": ["rail museum", "national rail museum", "railway museum"]
}

STOP_WORDS = {
    "what", "where", "how", "when", "why", "who", "which", "is", "are", "was", "were",
    "the", "a", "an", "in", "at", "near", "nearest", "closer", "closest", "next", "to", "for", "from",
    "around", "of", "and", "or", "but", "so", "with", "without", "between", "among", "along",
    "best", "good", "top", "famous", "popular", "great", "iconic", "also", "just",
    "cafe", "cafes", "restaurant", "restaurants", "food", "dining", "eat", "eateries", "eating",
    "metro", "station", "transit", "route", "reach", "get", "direction", "directions",
    "ticket", "tickets", "price", "prices", "cost", "fee", "fees", "entry", "fare",
    "timing", "timings", "hours", "time", "open", "closed", "closure", "schedule",
    "history", "heritage", "architecture", "story", "about", "tell", "me", "guide",
    "visit", "visiting", "tour", "explore", "trip", "see", "show", "know", "give", "find",
    "delhi", "new delhi", "dilli", "city", "place", "places", "spot", "spots",
    # Pronouns & follow-up reference tokens
    "it", "its", "there", "this", "that", "here", "their", "them", "these", "those",
    "i", "me", "my", "myself", "we", "us", "our", "you", "your", "yours", "he", "him", "his", "she", "her",
    "one", "can", "could", "would", "should", "does", "do", "did", "have", "has", "had",
    "any", "some", "all", "much", "many", "allowed", "allow", "rule", "rules", "guideline",
    "nearby", "around", "close", "local", "please", "suggest", "recommend", "recommendation",
    "recommendations", "option", "options", "anything", "something"
}


def get_known_corpus_places() -> dict[str, list[str]]:
    """Builds a dynamic lookup of known places from KNOWN_DELHI_PLACES, data/*.md, data/*.pdf, and workspace files."""
    corpus_places = dict(KNOWN_DELHI_PLACES)
    search_files = []
    data_dir = Path("data")
    if data_dir.exists():
        search_files.extend(data_dir.glob("*.md"))
        search_files.extend(data_dir.glob("*.pdf"))
    search_files.extend(Path(".").glob("*.pdf"))

    for p in search_files:
        stem = p.stem.lower()
        clean_stem = stem.replace(" ", "_")
        name = stem.replace("_", " ")

        # When registering stems from filenames like "akshardham_guide" or "jama_masjid_guide",
        # also strip trailing "_guide" so both "akshardham" and "akshardham_guide" map cleanly to the same place aliases!
        stripped_stem = re.sub(r"_guide$", "", clean_stem)
        stripped_name = stripped_stem.replace("_", " ")

        stems_to_register = {clean_stem, stripped_stem}
        aliases_to_add = [name, stripped_name, clean_stem, stripped_stem]

        for s in stems_to_register:
            if s not in corpus_places:
                corpus_places[s] = []
            for a in aliases_to_add:
                if a and a not in corpus_places[s]:
                    corpus_places[s].append(a)

    # Cross-link any _guide aliases with their base stem
    for place_k in list(corpus_places.keys()):
        if place_k.endswith("_guide"):
            base_k = place_k[:-6]
            if base_k in corpus_places:
                for a in corpus_places[base_k]:
                    if a not in corpus_places[place_k]:
                        corpus_places[place_k].append(a)
                for a in corpus_places[place_k]:
                    if a not in corpus_places[base_k]:
                        corpus_places[base_k].append(a)

    return corpus_places


def extract_target_location(query: str) -> str | None:
    """
    Extracts an open-ended target location phrase from any query.
    1. Looks for location prepositions (in, at, near, to, for, from, around, visit, reach, about).
    2. Strips standard question and category stop words.
    3. Returns cleaned target location phrase or None if query is purely general.
    """
    q = re.sub(r"[\?\,\.\!\;\:\'\"]", " ", query.lower())
    q = re.sub(r"\s+", " ", q).strip()

    # If the query contains pronoun references (there, it, here, etc.), it's a follow-up, not an open-ended location
    if re.search(r"\b(it|its|there|this|that|here)\b", q):
        return None

    # Preposition pattern
    prep_match = re.search(r"\b(?:in|at|near|to|for|from|around|visit|visiting|reach|about)\s+([a-zA-Z0-9\s\-]+)", q)
    if prep_match:
        phrase = prep_match.group(1).strip()
        phrase = re.sub(r"\b(by\s+(?:metro|bus|cab|car|auto|train|walk|foot))\b.*$", "", phrase).strip()
        words = phrase.split()
        cleaned_words = [w for w in words if w not in STOP_WORDS]
        if cleaned_words:
            candidate = " ".join(cleaned_words)
            if candidate not in ("delhi", "new delhi", "dilli") and len(candidate) >= 4:
                return candidate

    words = q.split()
    cleaned_words = [w for w in words if w not in STOP_WORDS]
    if cleaned_words:
        candidate = " ".join(cleaned_words)
        if candidate not in ("delhi", "new delhi", "dilli") and len(candidate) >= 4:
            return candidate

    return None


def detect_mentioned_places(query: str) -> list[str]:
    """
    Detects target locations using exact match, fuzzy SequenceMatcher (ratio >= 0.80),
    and open-ended location extraction.
    """
    q_lower = query.lower()
    corpus_places = get_known_corpus_places()
    detected = []

    # 1. Exact alias matches
    for place_canonical, aliases in corpus_places.items():
        if any(alias in q_lower for alias in aliases):
            detected.append(place_canonical)

    # 2. Open-ended extraction
    raw_loc = extract_target_location(query)

    # 3. Fuzzy matching against known corpus aliases (ratio >= 0.80)
    candidates_to_check = []
    if raw_loc:
        candidates_to_check.append(raw_loc)

    # Generate 1 to 4-word ngrams from query
    words = re.findall(r"[a-z0-9]+", q_lower)
    for n in range(1, 5):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram not in candidates_to_check and len(ngram) >= 4:
                candidates_to_check.append(ngram)

    for cand in candidates_to_check:
        for place_canonical, aliases in corpus_places.items():
            if place_canonical in detected:
                continue
            for alias in aliases:
                ratio = difflib.SequenceMatcher(None, cand, alias).ratio()
                if ratio >= 0.80:
                    detected.append(place_canonical)
                    break

    # If raw_loc was found and not yet in detected, append it as open-ended target
    if raw_loc and raw_loc not in detected:
        detected.append(raw_loc)

    return list(dict.fromkeys(detected))


def check_place_match(target_locations: list[str], metadata: dict) -> bool:
    """
    Verifies whether a chunk's metadata corresponds to any of the target locations.
    Checks substring matching and difflib.SequenceMatcher (ratio >= 0.75).
    """
    if not target_locations:
        return True

    place_id = metadata.get("place_id", "").lower().replace("-", "_")
    monument = metadata.get("monument", "").lower()
    corpus_places = get_known_corpus_places()

    for target in target_locations:
        t_clean = target.lower().replace("-", "_")
        # Direct substring
        if t_clean in place_id or place_id in t_clean or t_clean in monument:
            return True

        # Fuzzy match with chunk place_id or monument (ratio >= 0.75)
        if difflib.SequenceMatcher(None, t_clean, place_id).ratio() >= 0.75:
            return True
        if difflib.SequenceMatcher(None, t_clean, monument).ratio() >= 0.75:
            return True

        # Check aliases if target resolves to a known canonical place
        aliases = corpus_places.get(target, [])
        for alias in aliases:
            a_clean = alias.lower()
            if a_clean in place_id or a_clean in monument:
                return True
            if difflib.SequenceMatcher(None, a_clean, place_id).ratio() >= 0.75:
                return True
            if difflib.SequenceMatcher(None, a_clean, monument).ratio() >= 0.75:
                return True

    return False



# ==============================================================================
# STAGE 1: Zero-Latency Category Router
# ==============================================================================
def detect_query_categories(query: str) -> list[str]:
    """
    Inspects user query keywords to determine target metadata categories.

    If one or more categories match, returns the matched categories PLUS 'quick_facts'
    (as quick reference sections hold summary facts like metro/timing/closure).
    If no category matches, returns an empty list [] for unfiltered search.
    """
    q_lower = query.lower()
    matched = []

    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(cat)

    if matched:
        if "quick_facts" not in matched:
            matched.append("quick_facts")
        return matched

    return []


# ==============================================================================
# STAGE 2: Two-Tier ChromaDB Retrieval + Relevance Evaluator
# ==============================================================================
def retrieve_from_chromadb(
    query: str,
    top_k: int = 5,
    target_places: list[str] | None = None
) -> tuple[list[dict], str, dict]:
    """
    Two-Tier ChromaDB Retrieval + Corrective RAG Decision Gate.

    Args:
        query (str): The search query.
        top_k (int): Maximum chunks to retrieve (default: 5).
        target_places (list[str] | None): Optional inherited or known place identifiers.

    Returns:
        tuple[list[dict], str, dict]:
            - strong_chunks: filtered list of matching chunks
            - retrieval_status: 'LOCAL_HIT' | 'HYBRID_SUPPLEMENT' | 'WEB_FALLBACK' | 'EMPTY_DB'
            - diagnostics: dictionary containing routing metrics and candidate stats
    """
    total_docs = embedding.collection.count()
    detected_in_query = detect_mentioned_places(query)
    effective_places = list(dict.fromkeys((target_places or []) + detected_in_query))

    diagnostics = {
        "categories_detected": detect_query_categories(query),
        "total_docs_in_db": total_docs,
        "is_live_query": any(kw in query.lower() for kw in LIVE_KEYWORDS),
        "mentioned_places": effective_places,
        "candidates": [],
        "reason": ""
    }

    if total_docs == 0:
        diagnostics["reason"] = "Database is empty."
        return [], "EMPTY_DB", diagnostics

    # 1. Generate query embedding with task_type='RETRIEVAL_QUERY'
    query_vector = embedding.get_query_embedding(query)
    categories = diagnostics["categories_detected"]
    n_results = min(top_k, total_docs)
    candidates: dict[str, dict] = {}

    # Tier 1: Filtered query if categories detected
    if categories:
        try:
            where_clause = (
                {"category": categories[0]}
                if len(categories) == 1
                else {"category": {"$in": categories}}
            )
            tier1_res = embedding.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            if tier1_res and tier1_res.get("ids") and tier1_res["ids"][0]:
                for doc_id, text, meta, dist in zip(
                    tier1_res["ids"][0],
                    tier1_res["documents"][0],
                    tier1_res["metadatas"][0],
                    tier1_res["distances"][0]
                ):
                    dist_f = float(dist)
                    candidates[doc_id] = {
                        "id": doc_id,
                        "text": text,
                        "metadata": meta,
                        "distance": dist_f,
                        "similarity": round(1.0 - dist_f, 4),
                        "tier": "Tier 1 (Filtered)"
                    }
        except Exception:
            pass

    # Tier 2: Broad Unfiltered Fallback
    try:
        tier2_res = embedding.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        if tier2_res and tier2_res.get("ids") and tier2_res["ids"][0]:
            for doc_id, text, meta, dist in zip(
                tier2_res["ids"][0],
                tier2_res["documents"][0],
                tier2_res["metadatas"][0],
                tier2_res["distances"][0]
            ):
                dist_f = float(dist)
                if doc_id not in candidates or dist_f < candidates[doc_id]["distance"]:
                    candidates[doc_id] = {
                        "id": doc_id,
                        "text": text,
                        "metadata": meta,
                        "distance": dist_f,
                        "similarity": round(1.0 - dist_f, 4),
                        "tier": "Tier 2 (Broad)"
                    }
    except Exception:
        pass

    all_candidates = sorted(candidates.values(), key=lambda x: x["distance"])
    diagnostics["candidates"] = all_candidates

    # Corrective RAG Decision Gate:
    # 1. Similarity threshold: distance <= 0.48 (similarity >= 0.52)
    threshold_chunks = [c for c in all_candidates if c["distance"] <= 0.48]

    # 2. Place consistency verification:
    # If the user explicitly asks about a recognized place (e.g. Sunder Nursery, Lotus Temple),
    # verify that the retrieved chunks actually match that place!
    query_places = diagnostics["mentioned_places"]
    strong_chunks = []

    if query_places:
        for c in threshold_chunks:
            meta = c.get("metadata", {})
            if check_place_match(query_places, meta):
                strong_chunks.append(c)

        # Determine Routing Status for location-specific queries
        if not strong_chunks:
            retrieval_status = "WEB_FALLBACK"
            diagnostics["reason"] = f"Target location '{query_places[0].replace('_', ' ').title()}' is not currently in ChromaDB."
        elif diagnostics["is_live_query"]:
            retrieval_status = "HYBRID_SUPPLEMENT"
            diagnostics["reason"] = "Local chunks found, but user query requested live/real-time details."
        else:
            retrieval_status = "LOCAL_HIT"
            diagnostics["reason"] = f"Strong local match found ({len(strong_chunks)} chunks, similarity >= 52%)."
    else:
        # Fix 3: Query did not mention any specific place at all (e.g., "what are the best cafes in Delhi?")
        strong_chunks = threshold_chunks[:top_k]
        retrieval_status = "HYBRID_SUPPLEMENT"
        diagnostics["reason"] = "General Delhi query with no specific landmark. Local chunks supplemented with live city-wide Google Search."

    return strong_chunks[:top_k], retrieval_status, diagnostics


# ==============================================================================
# STAGE 3: Hybrid Answer Generation with Google Search Grounding
# ==============================================================================
SYSTEM_INSTRUCTION = """You are Dilli Darshan, a savvy, culturally immersed local guide for New Delhi. 
You are not a sterile search engine or Wikipedia scraper—you are an articulate, opinionated local companion who knows Delhi's unspoken rules, rhythm, sensory details, and transit hacks.

Key Guidelines:
1. Speak with Local Intuition: 
   - Weave verified operational facts (exact DMRC metro stations & exit gate numbers, ASI entry fees for Indians and foreigners, visiting hours, and weekly off days) seamlessly with practical local intuition.
   - Guide like an insider—the BEST time of day to arrive to beat long ticket queues or Delhi's blistering afternoon heat, which specific exit gate saves a 15-minute detour, which unauthorized e-rickshaw or guide traps to steer clear of, what signature dish to order nearby, and natural nearby next stops.
   - For dining, recommend signature dishes, vibe, and practical realities (cash-only spots, wait times, narrow alleys).
2. Deep Personalization & Physical/Dietary Care (Hindsight Memory Integration):
   - Strictly honor all remembered preferences, physical constraints, and dietary habits from Hindsight Cloud Agent Memory.
   - If the tourist has physical constraints (e.g., knee injury, elderly, difficulty climbing steep stairs, stroller or wheelchair access), proactively advise on accessible ramps, lift options, or alternate vantage points to avoid painful climbs (e.g. at monumental minarets, steep flight of stairs at high gates).
   - If the tourist is vegetarian, recommend iconic vegetarian spots (e.g., Haldiram's, Kake Di Hatti, Giani's, Paranthe Wali Gali) and do not steer them to non-vegetarian kebabs.
   - Tailor recommendations directly to their declared travel style (Heritage & History, Food & Street Markets, Budget & Metro Transit, Family & Leisure).
3. Grounded Authenticity & Secular Tone:
   - Strictly prioritize verified local guidebook context and live web details. Never invent ticket prices, timings, or transit routes.
   - Maintain a welcoming, inclusive, modern, and strictly secular tone. Do not use sectarian or religious greetings.
4. ZERO GREETINGS & DIRECTNESS (CRITICAL):
   - ABSOLUTELY NO GREETINGS OR SALUTATIONS: Never begin ANY response with greetings, salutations, pleasantries, or welcomes. Strictly FORBIDDEN at the start of any response: "Hello", "Hi", "Hey", "Greetings", "Welcome", "Good morning/afternoon", "Namaste", "Pranam", "As-salamu alaykum", "Sat Sri Akaal", or their equivalents in any language (such as "नमस्ते" or "हेलो").
   - NEVER OPEN BY ADDRESSING THE VISITOR BY NAME: Do not start responses with "Hello Saif,", "Saif,", "Dear visitor,", etc.
   - JUMP STRAIGHT TO THE POINT: Begin immediately and directly with the substantive answer, core facts, and recommendations from the very first word.
   - NO CONVERSATIONAL FILLER: Avoid introductory throat-clearing such as "Certainly! Here is...", "Sure thing!", "I would be happy to help with...", or "Great question!".
   - APPLIES TO EVERY TURN: This zero-greeting, straight-to-the-point rule strictly applies across the entire conversation—the very first question, second turn, third turn, follow-ups, and all subsequent messages alike.
5. Language Adaptability:
   - If Hindi is requested (Preferred Language: Hindi), respond in authentic, expressive, and respectful Hindi (Devanagari script) with accurate landmark and metro names. Start directly with the information without greeting words (do NOT start with 'नमस्ते', 'नमस्कार', or 'हेलो').
   - Otherwise, respond in natural, engaging, and professional English.
6. Clean Formatting & Source Attribution:
   - Format with clean Markdown using bold key facts for effortless reading.
   - Conclude your response with a concise `[Sources & Grounding]` line identifying the verified guidebook sections or live web sources used, so the companion UI can extract the source badge."""


def strip_leading_greetings(text: str, user_name: str | None = None) -> str:
    """
    Defensive safety net: strips any accidental leading greeting, pleasantry, or opening name call
    (e.g., 'Hello!', 'Hello Saif,', 'Hi Saif!', 'Namaste!', 'Hey there,', 'नमस्ते')
    so responses always jump straight to the substantive point.
    """
    if not text:
        return text

    cleaned = text.strip()

    # Build optional name component if user_name is known
    name_clause = ""
    if user_name and user_name.strip() and user_name.lower() not in ("tourist", "u", ""):
        name_clause = rf"(?:\s+{re.escape(user_name.strip())})?"

    greeting_pattern = re.compile(
        rf"^(?:(?:\*{{1,2}}|_{{1,2}})\s*)?(?:Hello|Hi|Hey|Greetings|Welcome|Namaste|Pranam|Good\s+(?:morning|afternoon|evening)|नमस्ते|नमस्कार|हेलो)\b{name_clause}(?:(?:\*{{1,2}}|_{{1,2}})\s*)?[\s,\.!\-:*_]+",
        flags=re.IGNORECASE
    )

    for _ in range(2):
        match = greeting_pattern.match(cleaned)
        if match:
            cleaned = cleaned[match.end():].lstrip()
        else:
            break

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned if cleaned else text


IGNORE_SEARCH_DOMAINS = (
    "youtube.com", "youtu.be", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "pinterest.com", "linkedin.com"
)


def tinyfish_live_search_and_fetch(query: str, max_urls: int = 3) -> dict:
    """
    Executes live web search via TinyFish Python SDK (tf_client.search.query)
    and extracts clean markdown content from top Delhi travel result pages
    (tf_client.fetch.get_contents).
    """
    if not embedding.tf_client:
        return {
            "query": query,
            "urls": [],
            "sources": [],
            "markdown_context": "",
            "success": False,
            "error": "TinyFish client not initialized (missing API key)."
        }

    clean_q = query.strip()
    if not any(term in clean_q.lower() for term in ("delhi", "new delhi", "dilli")):
        search_query = f"{clean_q} New Delhi India travel guide timings tickets metro"
    else:
        search_query = clean_q

    try:
        search_res = embedding.tf_client.search.query(
            search_query,
            location="New Delhi, India"
        )

        filtered_urls = []
        sources_meta = []
        for r in getattr(search_res, "results", []):
            url = getattr(r, "url", "")
            title = getattr(r, "title", "Delhi Travel Guide")
            if not url.startswith("http"):
                continue
            if any(d in url.lower() for d in IGNORE_SEARCH_DOMAINS):
                continue
            filtered_urls.append(url)
            sources_meta.append({"title": title, "uri": url})
            if len(filtered_urls) >= max_urls:
                break

        if not filtered_urls:
            return {
                "query": search_query,
                "urls": [],
                "sources": [],
                "markdown_context": "",
                "success": False,
                "error": "No accessible web pages found."
            }

        fetch_res = embedding.tf_client.fetch.get_contents(
            urls=filtered_urls,
            format="markdown",
            links=False,
            image_links=False
        )

        markdown_blocks = []
        for res in getattr(fetch_res, "results", []):
            u = getattr(res, "url", "")
            t = getattr(res, "title", "") or "Web Resource"
            text = getattr(res, "text", "") or ""
            if text:
                # Cap text to ~3500 chars per page to keep context high-density
                clean_text = text.strip()[:3500]
                markdown_blocks.append(f"### Source: {t} ({u})\n{clean_text}")

        combined_md = "\n\n---\n\n".join(markdown_blocks)
        return {
            "query": search_query,
            "urls": filtered_urls,
            "sources": sources_meta,
            "markdown_context": combined_md,
            "success": bool(combined_md)
        }
    except Exception as e:
        return {
            "query": search_query if "search_query" in locals() else clean_q,
            "urls": [],
            "sources": [],
            "markdown_context": "",
            "success": False,
            "error": str(e)
        }


def tinyfish_agent_live_check(query_or_place: str, target_url: str = "https://www.google.com") -> dict:
    """
    Deploys an autonomous TinyFish browser agent (tf_client.agent.run) to navigate
    live web pages, verify current timings, ticket fees, and live status.
    """
    if not embedding.tf_client:
        return {
            "status": "FAILED",
            "steps": 0,
            "result": "TinyFish API key not configured.",
            "success": False
        }

    goal = (
        f"Find up-to-date visitor information for '{query_or_place}' in New Delhi, India: "
        f"1. Current opening hours and weekly closing day. "
        f"2. Entry ticket fees for Indian nationals and foreign tourists. "
        f"3. Nearest Delhi Metro station, line color, and gate number. "
        f"4. Any live visitor notices or booking instructions."
    )

    try:
        agent_res = embedding.tf_client.agent.run(
            goal=goal,
            url=target_url
        )
        raw_result = getattr(agent_res, "result", "")
        if isinstance(raw_result, dict):
            result_text = raw_result.get("result", str(raw_result))
        else:
            result_text = str(raw_result)

        return {
            "status": getattr(agent_res, "status", "COMPLETED"),
            "steps": getattr(agent_res, "num_of_steps", 0),
            "result": result_text,
            "success": True
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "steps": 0,
            "result": str(e),
            "success": False
        }


def run_agent_workflow(goal_or_place: str):
    """Executes and formats a complete TinyFish autonomous browser agent run."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]🤖 Deploying TinyFish Autonomous Browser Agent[/bold cyan]\n"
            f"Target Task: [bold yellow]'{goal_or_place}'[/bold yellow]\n"
            f"[dim]The autonomous browser agent is navigating live web portals to extract real-time visitor intelligence...[/dim]",
            border_style="cyan"
        )
    )
    with console.status("[bold cyan]Agent navigating and extracting live web data...", spinner="dots"):
        res = tinyfish_agent_live_check(goal_or_place)

    if res.get("success"):
        console.print(
            Panel(
                Markdown(res.get("result", "")),
                title=f"[bold green]✓ TinyFish Browser Agent Report ({res.get('steps', 0)} steps)[/bold green]",
                subtitle=f"[dim]Status: {res.get('status', 'COMPLETED')}[/dim]",
                border_style="green",
                padding=(1, 2)
            )
        )
    else:
        console.print(f"[bold red]TinyFish Browser Agent failed:[/bold red] {res.get('result')}\n")


def execute_gemini_generation(
    contents: str | list[types.Content],
    enable_search: bool = False,
    temperature: float = 0.5
) -> tuple[types.GenerateContentResponse, str]:
    """
    Executes content generation using MODEL_ID (gemini-3.8-flash) as default,
    with graceful fallback to FALLBACK_MODEL_ID (gemini-3.6-flash) if 3.8 fails,
    and COUNTERMEASURE_MODEL_ID (gemini-2.5-flash) if both 3.8 and 3.6 fail.
    Supports either single-turn string or multi-turn list[types.Content].
    Response temperature is configured to 0.5 by default for human-like conversational warmth.
    """
    tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else None
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=temperature,
        tools=tools
    )

    models_to_try = [MODEL_ID, FALLBACK_MODEL_ID, COUNTERMEASURE_MODEL_ID, "gemini-flash-latest"]
    last_err = None

    for idx, m in enumerate(models_to_try):
        try:
            response = embedding.client.models.generate_content(
                model=m,
                contents=contents,
                config=config
            )
            return response, m
        except Exception as e:
            last_err = e
            if idx < len(models_to_try) - 1:
                next_model = models_to_try[idx + 1]
                err_msg = str(e).strip()
                if len(err_msg) > 100:
                    err_msg = err_msg[:100] + "..."
                try:
                    console.print(
                        f"[yellow]⚠️  [Model Fallback][/yellow] Generation failed on [cyan]{m}[/cyan]: [dim]{err_msg}[/dim]\n"
                        f"   [bold green]↳ Automatically switching to '{next_model}'...[/bold green]"
                    )
                except Exception:
                    pass
            continue

    raise RuntimeError(f"All Gemini generation models failed ({', '.join(models_to_try)}). Last error: {last_err}")


def generate_dilli_darshan_answer(query: str, chunks: list[dict], status: str) -> dict:
    """
    Generates a grounded response using local context and/or live TinyFish web research,
    with graceful fallback to Gemini Google Search Grounding if necessary.
    """
    if chunks:
        blocks = []
        for idx, c in enumerate(chunks, start=1):
            meta = c.get("metadata", {})
            monument = meta.get("monument", "Delhi Monument")
            section = meta.get("section", "General")
            subsec = meta.get("subsection", "")
            sec_display = f"{section} > {subsec}" if subsec else section
            sim = c.get("similarity", round(1.0 - c.get("distance", 0.0), 4))
            chunk_id = c.get("id", f"c_{idx}")

            blocks.append(
                f"[Chunk {idx} | ID: {chunk_id} | Monument: {monument} | Section: {sec_display} | Similarity: {sim:.2%}]\n"
                f"{c.get('text', '').strip()}"
            )
        local_context = "\n\n".join(blocks)
    else:
        local_context = "No verified local guidebook chunks available for this location."

    tf_data = None
    live_engine = "Local ChromaDB Vector RAG"
    web_queries_used = []
    web_sources = []

    # If web grounding is needed, prioritize TinyFish live search & fetch
    if status in ("WEB_FALLBACK", "HYBRID_SUPPLEMENT", "EMPTY_DB"):
        tf_data = tinyfish_live_search_and_fetch(query, max_urls=3)
        if tf_data.get("success") and tf_data.get("markdown_context"):
            live_engine = "TinyFish Python SDK (Active Live Search & Markdown Fetch)"
            web_sources = tf_data.get("sources", [])
            web_queries_used = [tf_data.get("query", query)]

            prompt = f"""Verified Local Guidebook Context:
{local_context}

Verified Live Web Research Context (via TinyFish Live Web Engine):
{tf_data['markdown_context']}

User Question: {query}
Retrieval Mode: {status}

Provide an accurate, secular, professional, and practical guide response. Do NOT include any greeting or pleasantry (strictly forbid "Hello", "Hi", etc.)—jump immediately and directly into the answer. Base your facts on the verified local and live web research context:"""

            response, model_used = execute_gemini_generation(prompt, enable_search=False)
            answer_text = strip_leading_greetings(response.text or "")

            return {
                "answer": answer_text,
                "web_queries": web_queries_used,
                "web_sources": web_sources,
                "model_used": model_used,
                "status": status,
                "chunks": chunks,
                "tf_data": tf_data,
                "live_engine": live_engine
            }

        # Fallback to native Gemini Google Search Grounding if TinyFish returned no markdown
        live_engine = "Gemini Google Search Grounding (Fallback)"

    # Standard / Fallback Generation
    prompt = f"""Verified Local Guidebook Context:
{local_context}

User Question: {query}
Retrieval Mode: {status}

Provide an accurate, secular, professional, and practical guide response. Do NOT include any greeting or pleasantry (strictly forbid "Hello", "Hi", etc.)—jump immediately and directly into the answer:"""

    enable_search = status in ("WEB_FALLBACK", "HYBRID_SUPPLEMENT", "EMPTY_DB")
    response, model_used = execute_gemini_generation(prompt, enable_search=enable_search)
    answer_text = strip_leading_greetings(response.text or "")

    if response.candidates and response.candidates[0].grounding_metadata:
        gm = response.candidates[0].grounding_metadata
        if gm.web_search_queries:
            web_queries_used = list(gm.web_search_queries)
        if gm.grounding_chunks:
            for gc in gm.grounding_chunks:
                if getattr(gc, "web", None):
                    web_sources.append({
                        "title": gc.web.title or "Live Web Resource",
                        "uri": gc.web.uri or ""
                    })

    return {
        "answer": answer_text,
        "web_queries": web_queries_used,
        "web_sources": web_sources,
        "model_used": model_used,
        "status": status,
        "chunks": chunks,
        "tf_data": tf_data,
        "live_engine": live_engine
    }


# Backward-compatible alias
generate_dilliguide_answer = generate_dilli_darshan_answer


class DilliDarshanChatSession:
    """
    Stateful Multi-Turn Chat Session for Dilli Darshan.
    Integrates Hindsight Cloud as the unified conversational memory and user preference engine.
    Maintains contextual active place tracking and inherits monuments across follow-up queries.
    """
    def __init__(self, max_history_turns: int = 10):
        # Conversational memory and user preferences are managed directly by Hindsight Cloud
        self.last_active_places: list[str] = []
        self.turn_count: int = 0
        self.max_history_turns: int = max_history_turns

    def reset_session(self):
        """Clears active context place and resets turn counter."""
        self.last_active_places.clear()
        self.turn_count = 0

    def resolve_contextual_query(
        self,
        raw_query: str,
        recalled_memories: list[str] | None = None
    ) -> tuple[str, list[str], bool]:
        """
        Determines the effective query and target places.
        1. If explicit places are found in raw_query, updates last_active_places.
        2. If last_active_places is empty (e.g. server restart), scans recalled
           Hindsight Cloud memories for any known Delhi monument from KNOWN_DELHI_PLACES
           and inherits it automatically.
        3. If last_active_places is non-empty, detects follow-up intent (pronouns,
           travel planning keywords) and inherits the active monument for ChromaDB.

        Returns:
            tuple[str, list[str], bool]: (effective_query, active_places, is_followup)
        """
        detected_places = detect_mentioned_places(raw_query)
        corpus = get_known_corpus_places()
        known_detected = [p for p in detected_places if p in corpus]

        # 1. Explicit known place(s) mentioned in current turn
        if known_detected:
            self.last_active_places = known_detected
            return raw_query, known_detected, False

        # 2. Inherit from recalled Hindsight memories if last_active_places is currently empty
        if not self.last_active_places and recalled_memories:
            for mem_text in recalled_memories:
                places_in_mem = detect_mentioned_places(mem_text)
                known_mem = [p for p in places_in_mem if p in corpus]
                if known_mem:
                    self.last_active_places = known_mem
                    break
                # Fallback alias search in memory text
                mem_low = mem_text.lower()
                matched = False
                for place_k, aliases in corpus.items():
                    if any(alias in mem_low for alias in aliases):
                        self.last_active_places = [place_k]
                        matched = True
                        break
                if matched:
                    break

        # 3. Follow-Up Resolution: If last_active_places is non-empty
        if self.last_active_places:
            q_lower = raw_query.lower()

            # Pronoun / reference triggers
            has_pronoun = bool(
                re.search(r"\b(it|its|there|this place|that place|that|here|their|them|the place)\b", q_lower)
            )

            # Category question / travel planning question triggers
            has_category_kw = bool(detect_query_categories(raw_query))
            has_travel_kw = any(w in q_lower for w in [
                "timing", "ticket", "fee", "cost", "metro", "reach", "food", "cafe",
                "photo", "camera", "allowed", "dress", "rule", "open", "close", "time",
                "nearest", "gate", "distance", "worth", "visit", "architecture", "history", "guide",
                "eat", "lunch", "dinner", "breakfast", "stairs", "climb", "walk", "wheelchair", "injury"
            ])

            # Prevent follow-up hijacking on explicit city-wide queries (e.g. "in Delhi", "across New Delhi")
            if not has_pronoun and re.search(r"\b(in|across|around|all over)\s+(new\s+)?delhi\b", q_lower):
                return raw_query, [], False

            if has_pronoun or has_category_kw or has_travel_kw:
                # Inherit active monument from previous turn or Hindsight Cloud memory
                active_key = self.last_active_places[0]
                active_title = active_key.replace("_", " ").title()
                if active_key in corpus and corpus[active_key]:
                    candidate = corpus[active_key][0].title()
                    if len(candidate) > 3:
                        active_title = candidate

                resolved_search_query = f"{active_title}: {raw_query}"
                return resolved_search_query, self.last_active_places, True

        # 4. Open-ended place candidate (if not a follow-up and detected_places has open-ended guess)
        if detected_places:
            self.last_active_places = detected_places
            return raw_query, detected_places, False

        return raw_query, [], False

    def send_message(
        self,
        raw_query: str,
        user_id: int | None = None,
        user_name: str = "Tourist",
        travel_style: str = "Heritage & History",
        language: str = "English",
        user_context: dict | None = None,
        **kwargs
    ) -> dict:
        """
        Processes a single user turn in the multi-turn session using Hindsight Cloud:
        Step 1: Recall user memories and preferences from Hindsight Cloud.
        Step 2: Resolve contextual query and place inheritance.
        Step 3: Retrieve local guidebook chunks from ChromaDB and live web context from TinyFish if needed.
        Step 4: Construct grounded turn prompt combining ChromaDB, TinyFish, and Hindsight Cloud Agent Memory.
        Step 5: Generate answer via Gemini with True Local Companion Persona.
        Step 6: Retain chat turn in Hindsight Cloud asynchronously with zero latency.
        """
        # Backward-compatibility unpack if user_context dict is passed
        if user_context:
            user_id = user_id or user_context.get("user_id") or user_context.get("id")
            user_name = user_context.get("user_name") or user_context.get("name") or user_name
            travel_style = user_context.get("travel_style") or user_context.get("style") or travel_style
            language = user_context.get("language") or user_context.get("lang") or language

        # Increment multi-turn counter
        self.turn_count += 1

        # Step 1: Call Hindsight Cloud recall
        recalled = memory.recall_user_memories(
            user_id=user_id,
            query=raw_query,
            user_name=user_name
        )
        recalled_memories = recalled.get("memories", [])
        formatted_memory_context = recalled.get("formatted_memory_context", "")

        # Step 2: Contextual query resolution (place inheritance via memory if needed)
        effective_query, active_places, is_followup = self.resolve_contextual_query(
            raw_query,
            recalled_memories=recalled_memories
        )

        # Step 3: Retrieve from ChromaDB with active_places passed for verification
        chunks, status, diagnostics = retrieve_from_chromadb(effective_query, target_places=active_places)

        # Format local guidebook chunks
        if chunks:
            blocks = []
            for idx, c in enumerate(chunks, start=1):
                meta = c.get("metadata", {})
                monument = meta.get("monument", "Delhi Monument")
                section = meta.get("section", "General")
                subsec = meta.get("subsection", "")
                sec_display = f"{section} > {subsec}" if subsec else section
                sim = c.get("similarity", round(1.0 - c.get("distance", 0.0), 4))
                chunk_id = c.get("id", f"c_{idx}")

                blocks.append(
                    f"[Chunk {idx} | ID: {chunk_id} | Monument: {monument} | Section: {sec_display} | Similarity: {sim:.2%}]\n"
                    f"{c.get('text', '').strip()}"
                )
            local_context = "\n\n".join(blocks)
        else:
            local_context = "No verified local guidebook chunks available for this location."

        tf_data = None
        live_engine = "Local ChromaDB Vector RAG"
        web_queries_used = []
        web_sources = []

        # Step 3 (cont): Live web research if needed
        if status in ("WEB_FALLBACK", "HYBRID_SUPPLEMENT", "EMPTY_DB"):
            tf_data = tinyfish_live_search_and_fetch(effective_query, max_urls=3)
            if tf_data.get("success") and tf_data.get("markdown_context"):
                live_engine = "TinyFish Python SDK (Active Live Search & Markdown Fetch)"
                web_sources = tf_data.get("sources", [])
                web_queries_used = [tf_data.get("query", effective_query)]

        # Step 4: Construct Grounded Turn Prompt combining:
        # - Verified Local Guidebook Context
        # - Verified Live Web Research Context (TinyFish)
        # - Hindsight Cloud Agent Memory (User Login Profile, Learned Preferences & Past Context)
        # - User Question
        place_label = ", ".join(p.replace("_", " ").title() for p in active_places) if active_places else "General Delhi"

        memory_block_parts = [
            f"- User Profile: Name: {user_name} | Travel Style: {travel_style} | Preferred Language: {language}"
        ]
        if formatted_memory_context:
            memory_block_parts.append(formatted_memory_context)
        else:
            memory_block_parts.append("- No previous interaction memories recorded yet.")

        memory_prompt_block = (
            "Hindsight Cloud Agent Memory (User Login Profile, Learned Preferences & Past Context):\n"
            + "\n".join(memory_block_parts) + "\n\n"
            "CRITICAL INSTRUCTION FOR MEMORY INTEGRATION:\n"
            "Strictly respect and weave all user constraints, accessibility needs (e.g. knee injuries, stairs avoidance, elevator/ramps), "
            "dietary restrictions (e.g. strict vegetarian spots, avoidance of non-veg), budget, and travel preferences into your answer! "
            "Never recommend activities, stairs, or dining that contradict the tourist's profile or remembered preferences."
        )

        prompt_elements = [
            f"Verified Local Guidebook Context:\n{local_context}"
        ]
        if tf_data and tf_data.get("markdown_context"):
            prompt_elements.append(
                f"Verified Live Web Research Context (via TinyFish Live Web Engine):\n{tf_data['markdown_context']}"
            )

        prompt_elements.extend([
            memory_prompt_block,
            f"User Question: {raw_query}",
            f"Active Target Landmark: {place_label}",
            f"Retrieval Mode: {status}",
            "Provide an accurate, secular, culturally grounded, insider guide response. Weave verified operational facts (metro gates, ticket fees, timings) with practical local intuition. Strictly obey the zero-greeting directive—begin immediately with the substantive answer:"
        ])

        grounded_turn_prompt = "\n\n".join(prompt_elements)

        # Step 5: Generate answer via Gemini
        enable_search = (status in ("WEB_FALLBACK", "HYBRID_SUPPLEMENT", "EMPTY_DB")) and not (tf_data and tf_data.get("markdown_context"))
        response, model_used = execute_gemini_generation(contents=grounded_turn_prompt, enable_search=enable_search)
        answer_text = response.text or ""

        # Extract Google Search Grounding metadata if fallback was triggered
        if response.candidates and response.candidates[0].grounding_metadata:
            gm = response.candidates[0].grounding_metadata
            if gm.web_search_queries and not web_queries_used:
                web_queries_used = list(gm.web_search_queries)
                live_engine = "Gemini Google Search Grounding (Fallback)"
            if gm.grounding_chunks and not web_sources:
                for gc in gm.grounding_chunks:
                    if getattr(gc, "web", None):
                        web_sources.append({
                            "title": gc.web.title or "Live Web Resource",
                            "uri": gc.web.uri or ""
                        })

        # Determine concise source badge string for frontend pill
        if status == "LOCAL_HIT" and chunks:
            monuments = []
            for c in chunks:
                m = c.get("metadata", {}).get("monument")
                if m and m not in monuments:
                    monuments.append(m)
            if monuments:
                clean_names = " & ".join(monuments[:2])
                source_badge_text = f"{clean_names} Archives (ChromaDB) + Hindsight Cloud Memory"
            else:
                source_badge_text = "Delhi Tourism Archives (ChromaDB) + Hindsight Cloud Memory"
        elif status == "HYBRID_SUPPLEMENT":
            source_badge_text = "ChromaDB + TinyFish Live Web + Hindsight Cloud Memory"
        elif status in ("WEB_FALLBACK", "EMPTY_DB"):
            if "TinyFish" in live_engine:
                source_badge_text = "TinyFish Live Web + Hindsight Cloud Memory"
            else:
                source_badge_text = "Gemini Live Web + Hindsight Cloud Memory"
        else:
            source_badge_text = "Delhi Tourism Guide + Hindsight Cloud Memory"

        # Clean any raw [Sources & Grounding] footer from the main answer body
        clean_answer = re.sub(
            r"\n*(?:\[Sources\s*&\s*Grounding\]|###?\s*Sources(?:\s*&\s*Grounding)?|---+\s*\n\s*\*\*Sources?\*\*:?).*$",
            "",
            answer_text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        if not clean_answer:
            clean_answer = answer_text

        # Defensive safety net: guarantee leading greetings are removed
        clean_answer = strip_leading_greetings(clean_answer, user_name)

        # Step 6: Retain chat turn in Hindsight Cloud asynchronously
        memory.retain_chat_turn(
            user_id=user_id,
            user_name=user_name,
            user_query=raw_query,
            bot_answer=clean_answer,
            active_places=active_places
        )

        return {
            "answer": clean_answer,
            "raw_answer": answer_text,
            "source": source_badge_text,
            "source_badge_text": source_badge_text,
            "web_queries": web_queries_used,
            "web_sources": web_sources,
            "model_used": model_used,
            "status": status,
            "chunks": chunks,
            "tf_data": tf_data,
            "live_engine": live_engine,
            "diagnostics": diagnostics,
            "effective_query": effective_query,
            "is_followup": is_followup,
            "active_places": active_places,
            "turn_number": self.turn_count,
            "hindsight_bank_id": recalled.get("bank_id"),
            "memories_recalled": recalled_memories,
            "memories_recalled_count": len(recalled_memories)
        }


# ==============================================================================
# STAGE 4: Self-Learning Cache
# ==============================================================================
def auto_cache_to_chromadb(place_name: str, grounded_markdown: str) -> tuple[bool, int, str]:
    """
    Saves a structured Markdown guidebook to data/<place_id>.md, chunks it,
    and upserts into ChromaDB.

    Returns:
        tuple[bool, int, str]: (success, chunk_count, file_path)
    """
    clean_place_id = re.sub(r"[^a-z0-9_]", "", place_name.lower().replace(" ", "_").replace("-", "_")).strip("_")
    if not clean_place_id:
        clean_place_id = "delhi_monument"

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    target_file = data_dir / f"{clean_place_id}.md"

    # Write Markdown file
    target_file.write_text(grounded_markdown, encoding="utf-8")

    # Chunk the Markdown file
    chunks = chunking.chunk_markdown_text(
        full_markdown=grounded_markdown,
        source_name=target_file.name,
        place_id=clean_place_id
    )

    if not chunks:
        return False, 0, str(target_file)

    # Persist into ChromaDB
    doc_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]

    embedding.add_documents(doc_ids=doc_ids, texts=texts, metadatas=metas)
    return True, len(chunks), str(target_file)


def auto_cache_new_place(query: str, answer_text: str) -> tuple[bool, str, int]:
    """
    Formats the grounded web information into a standardized 5-section Markdown
    guidebook and persists it into ChromaDB. Uses TinyFish to fetch any missing
    context so all 5 sections are 100% complete without placeholders.
    """
    query_places = detect_mentioned_places(query)
    if query_places:
        place_name = query_places[0].replace("_", " ").title()
    else:
        target_loc = extract_target_location(query)
        place_name = target_loc.replace("_", " ").title() if target_loc else "Delhi Landmark"

    # Fetch comprehensive context via TinyFish to ensure 100% complete coverage
    research_query = f"{place_name} New Delhi history architecture timings entry ticket fee nearest metro station iconic food cafes"
    tf_data = tinyfish_live_search_and_fetch(research_query, max_urls=3)
    comprehensive_context = tf_data.get("markdown_context", "")

    structuring_prompt = f"""You are an expert travel editor for Dilli Darshan.
Create our standardized, high-quality 5-section Dilli Darshan guidebook for: {place_name}.

Initial Information:
{answer_text}

Additional Live Web Research Context (via TinyFish Live Web Engine):
{comprehensive_context}

CRITICAL QUALITY & FACTUALITY REQUIREMENTS:
1. EVERY SINGLE SECTION (1 through 5) MUST BE THOROUGHLY AND COMPLETELY FILLED with verified, factual travel information.
2. NEVER output placeholder text such as "*Information is not available in the provided context*", "N/A", or "Details not provided". Use the provided live web research context to detail the history, architecture, timings, tickets, metro route, eateries, and visitor tips.
3. Keep a strictly secular and professional tone (no religious salutations like "Namaste").

Output Structure:
# {place_name}
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
        response, _ = execute_gemini_generation(structuring_prompt, enable_search=False, temperature=0.2)
        structured_md = (response.text or "").strip()
        if structured_md.startswith("```markdown"):
            structured_md = structured_md[len("```markdown"):].strip()
        if structured_md.startswith("```"):
            structured_md = structured_md[len("```"):].strip()
        if structured_md.endswith("```"):
            structured_md = structured_md[:-3].strip()

        # Extract place title from first line
        first_line = structured_md.split("\n")[0]
        parsed_title = re.sub(r"^#+\s*", "", first_line).strip() or place_name

        success, count, path_saved = auto_cache_to_chromadb(parsed_title, structured_md)
        return success, path_saved, count
    except Exception as e:
        console.print(f"[bold red][Error in Auto-Cache]:[/bold red] {e}")
        return False, "", 0


# ==============================================================================
# STAGE 5: Interactive Terminal Diagnostics & CLI
# ==============================================================================
def display_diagnostic_panel(
    query: str,
    status: str,
    diagnostics: dict,
    result: dict,
    chat_session: "DilliDarshanChatSession | None" = None
):
    """Renders the comprehensive Terminal Diagnostic Panel before displaying answer."""
    categories = diagnostics.get("categories_detected", [])
    cats_display = f"[bold cyan]{categories}[/bold cyan]" if categories else "[dim]Unfiltered Search[/dim]"

    if status == "LOCAL_HIT":
        decision_style = "bold green"
        decision_label = "LOCAL_HIT  (⚡ Fast Local RAG - High Confidence)"
    elif status == "HYBRID_SUPPLEMENT":
        decision_style = "bold yellow"
        decision_label = "HYBRID_SUPPLEMENT  (🔄 Local Guidebook + Live Web Research)"
    elif status == "EMPTY_DB":
        decision_style = "bold red"
        decision_label = "EMPTY_DB  (⚠️ Local ChromaDB is Empty -> Live Web Research)"
    else:
        decision_style = "bold magenta"
        decision_label = "WEB_FALLBACK  (🌐 Live TinyFish Web Search Triggered!)"

    model_used = result.get("model_used", MODEL_ID)
    live_engine = result.get("live_engine", "Local ChromaDB Vector RAG")
    tf_data = result.get("tf_data")
    web_queries = result.get("web_queries", [])

    # Multi-turn session context diagnostics
    turn_num = result.get("turn_number") or (getattr(chat_session, "turn_count", 1) if chat_session else 1)
    active_places = result.get("active_places") or (chat_session.last_active_places if chat_session else diagnostics.get("mentioned_places", []))
    memories_recalled_count = result.get("memories_recalled_count", 0)
    is_followup = result.get("is_followup", False)
    effective_query = result.get("effective_query", query)

    console.print()
    console.print("=" * 80, style="cyan")
    console.print(f"             [bold yellow]🧭 Dilli Darshan Terminal Diagnostic Panel[/bold yellow]", justify="center")
    console.print("=" * 80, style="cyan")

    console.print(f"• [bold]User Query:[/bold] {query}")
    console.print(f"• [bold]Session Turn & Memory:[/bold] Turn #{turn_num} | Active Context Place: {active_places} | Hindsight Memories Recalled: {memories_recalled_count}")
    if is_followup:
        console.print(f"• [bold]Resolved Search Query:[/bold] [bold yellow]\"{effective_query}\"[/bold yellow] [dim](Inherited from chat context)[/dim]")
    console.print(f"• [bold]Query Categories Detected:[/bold] {cats_display}")
    console.print(f"• [bold]Routing Decision:[/bold] [{decision_style}]{decision_label}[/{decision_style}]")
    console.print(f"• [bold]Routing Rationale:[/bold] [italic]{diagnostics.get('reason', '')}[/italic]")
    console.print(f"• [bold]Live Grounding Engine:[/bold] [bold green]{live_engine}[/bold green]")
    console.print(f"• [bold]Active Model:[/bold] [green]{model_used}[/green] [dim](Temperature: 0.5)[/dim]")

    candidates = diagnostics.get("candidates", [])
    if candidates:
        console.print("\n[bold]ChromaDB Candidate Evaluation:[/bold]")
        table = Table(box=None, header_style="bold cyan", pad_edge=False)
        table.add_column("#", justify="center", width=3)
        table.add_column("Chunk ID", style="dim", width=16)
        table.add_column("Monument", style="bold green", max_width=22, overflow="ellipsis")
        table.add_column("Category", style="magenta", max_width=24)
        table.add_column("Similarity", justify="right", style="yellow", width=11)
        table.add_column("Gate Status", style="white")

        for idx, c in enumerate(candidates[:5], start=1):
            meta = c.get("metadata", {})
            monument = meta.get("monument", "Delhi")
            category = meta.get("category", "general")
            sim = c.get("similarity", 0.0)
            dist = c.get("distance", 1.0)
            
            # Diagnostic gate evaluation note
            if dist > 0.48:
                gate_status = "[red]Below Threshold (< 52%)[/red]"
            elif diagnostics.get("mentioned_places") and not check_place_match(diagnostics["mentioned_places"], meta):
                gate_status = "[yellow]Place Mismatch[/yellow]"
            elif not diagnostics.get("mentioned_places"):
                gate_status = "[cyan]General City (Supplemented)[/cyan]"
            else:
                gate_status = "[green]✓ Selected (Strong)[/green]"

            table.add_row(str(idx), c.get("id", ""), monument, category, f"{sim:.2%}", gate_status)

        console.print(table)
    else:
        console.print("\n[dim]No local candidates retrieved from ChromaDB.[/dim]")

    if tf_data and tf_data.get("urls"):
        console.print(f"\n[bold green]TinyFish Live Web Research Engine:[/bold green]")
        console.print(f"  • Search Query: [italic]{tf_data.get('query', '')}[/italic]")
        console.print(f"  • Pages Extracted ({len(tf_data.get('urls', []))}):")
        for s in tf_data.get("sources", []):
            console.print(f"    - [blue underline]{s['title']}[/blue underline]: [dim]{s['uri']}[/dim]")
    elif web_queries:
        console.print("\n[bold cyan]Google Search Grounding Queries Executed (Fallback):[/bold cyan]")
        for q in web_queries:
            console.print(f"  • [italic]{q}[/italic]")

    console.print("=" * 80, style="cyan")
    console.print()


def show_session_history(chat_session: DilliDarshanChatSession):
    """Displays a clean summary of current chat session and active place."""
    if not chat_session.last_active_places:
        console.print("\n[yellow]No active monument or landmark in the current session yet.[/yellow]\n")
        return
    places_str = ", ".join(p.replace("_", " ").title() for p in chat_session.last_active_places)
    console.print(f"\n[bold cyan]🏛️ Active Delhi Landmark:[/bold cyan] [green]{places_str}[/green]\n")


def run_query_pipeline(
    query: str,
    chat_session: DilliDarshanChatSession | None = None,
    auto_cache_enabled: bool = False
):
    """Executes the complete retrieval, generation, and caching pipeline for a query."""
    if chat_session:
        with console.status("[bold cyan]Evaluating context & generating answer with Dilli Darshan...", spinner="dots"):
            result = chat_session.send_message(query)
        status = result.get("status", "LOCAL_HIT")
        diagnostics = result.get("diagnostics", {})
        effective_query = result.get("effective_query", query)
    else:
        with console.status("[bold cyan]Evaluating ChromaDB & routing query...", spinner="dots"):
            chunks, status, diagnostics = retrieve_from_chromadb(query)

        with console.status("[bold cyan]Generating answer with Gemini & Grounding...", spinner="dots"):
            result = generate_dilli_darshan_answer(query, chunks, status)
        effective_query = query

    display_diagnostic_panel(
        query=query,
        status=status,
        diagnostics=diagnostics,
        result=result,
        chat_session=chat_session
    )

    # Display Answer
    console.print(
        Panel(
            Markdown(result["answer"]),
            title="[bold green]🏛️  Dilli Darshan Response[/bold green]",
            subtitle=f"[dim]Mode: {status} | Model: {result['model_used']}[/dim]",
            border_style="green",
            padding=(1, 2)
        )
    )

    # Display Web Sources if present
    if result.get("web_sources"):
        console.print("\n[bold yellow]Live Grounding Sources:[/bold yellow]")
        for s in result["web_sources"][:4]:
            console.print(f"  • [blue underline]{s['title']}[/blue underline]: [dim]{s['uri']}[/dim]")

    # Self-Learning Cache Trigger
    if status in ("WEB_FALLBACK", "EMPTY_DB"):
        console.print()
        should_cache = auto_cache_enabled
        if not auto_cache_enabled:
            try:
                user_choice = console.input("[bold yellow]💡 This place wasn't in your local ChromaDB! Save this verified guide to ChromaDB for instant future retrieval? (y/N): [/bold yellow]").strip().lower()
                should_cache = user_choice in ("y", "yes")
            except (KeyboardInterrupt, EOFError):
                should_cache = False

        if should_cache:
            with console.status("[bold cyan]Structuring and saving 100% complete guide to ChromaDB cache...", spinner="dots"):
                ok, saved_path, count = auto_cache_new_place(effective_query, result["answer"])
            if ok:
                console.print(f"[bold green]✓ Successfully cached![/bold green] Saved [yellow]{saved_path}[/yellow] ({count} chunks added to ChromaDB).")
                console.print(f"[dim]ChromaDB total now: {embedding.collection.count()} chunks.[/dim]\n")
            else:
                console.print("[bold red]Failed to auto-cache place to ChromaDB.[/bold red]\n")


def print_banner():
    total_docs = embedding.collection.count()
    backup_info = f" (Backups: {embedding.client.backup_label})" if getattr(embedding.client, "backup_label", None) else ""
    console.print(
        Panel.fit(
            f"[bold cyan]Connected to ChromaDB:[/bold cyan]    [green]{embedding.collection.name}[/green] ([yellow]{total_docs} chunks[/yellow])\n"
            f"[bold cyan]Gemini Client Key:[/bold cyan]        [green]{getattr(embedding.client, 'active_label', 'API_Key')}[/green][dim]{backup_info}[/dim]\n"
            f"[bold cyan]Active Generation Model:[/bold cyan]  [green]{MODEL_ID}[/green] (Fallback: [dim]{FALLBACK_MODEL_ID} -> {COUNTERMEASURE_MODEL_ID}[/dim])\n"
            f"[bold cyan]Multi-Turn Chat Session:[/bold cyan]  [bold green]Stateful Context & Memory Active[/bold green]\n"
            f"[bold cyan]Live Search & Extraction:[/bold cyan] [bold green]TinyFish Python SDK[/bold green] (Fallback: Google Grounding)\n"
            f"[bold cyan]Autonomous Browser Agent:[/bold cyan] [bold green]TinyFish Agent Ready[/bold green] (Type '/agent <task>')\n"
            f"[bold cyan]Self-Learning Cache:[/bold cyan]     [bold green]READY[/bold green] (Auto-creates 100% complete .md & embeds)\n\n"
            "[dim]Commands: Type query, 'history' (/history), 'new' (/new, resetchat), '/agent <task>', '--auto-cache', 'stats', 'clear', or 'exit'.[/dim]",
            title="[bold yellow]🏛️  Dilli Darshan - Stateful Multi-Turn Retrieval & Generation Engine[/bold yellow]",
            border_style="cyan",
            padding=(1, 2)
        )
    )


def show_stats():
    total_docs = embedding.collection.count()
    console.print(f"\n[bold cyan]Collection Name:[/bold cyan] {embedding.collection.name}")
    console.print(f"[bold cyan]Total Chunks in DB:[/bold cyan] {total_docs}")
    if total_docs > 0:
        try:
            sample = embedding.collection.get(include=["metadatas"])
            monuments = sorted(set(m.get("monument", "Unknown") for m in sample["metadatas"] if m))
            console.print("[bold cyan]Monuments / Guides in Knowledge Base:[/bold cyan]")
            for m in monuments:
                console.print(f"  • {m}")
        except Exception:
            pass
    console.print()


def main():
    print_banner()

    chat_session = DilliDarshanChatSession()

    # CLI argument mode: Check for --agent / -ag
    if "--agent" in sys.argv or "-ag" in sys.argv:
        flag = "--agent" if "--agent" in sys.argv else "-ag"
        flag_idx = sys.argv.index(flag)
        agent_query_parts = [a for idx, a in enumerate(sys.argv) if idx > flag_idx and not a.startswith("--")]
        if agent_query_parts:
            agent_goal = " ".join(agent_query_parts).strip()
            run_agent_workflow(agent_goal)
            return
        else:
            console.print("[bold red]Please specify a goal or question for the browser agent, e.g.:[/bold red] python retrieval.py --agent \"Check Lotus Temple timings\"")
            return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto_cache_flag = "--auto-cache" in sys.argv or "-a" in sys.argv

    if args:
        query = " ".join(args).strip()
        if query.lower() == "stats":
            show_stats()
            return
        elif query.lower() in ("history", "/history"):
            show_session_history(chat_session)
            return
        elif query.lower() in ("clear", "reset"):
            confirm = console.input("[bold red]Are you sure you want to clear ChromaDB? (y/N): [/bold red]").strip().lower()
            if confirm in ("y", "yes"):
                embedding.clear_database()
            return
        run_query_pipeline(query, chat_session=chat_session, auto_cache_enabled=auto_cache_flag)
        return

    # Interactive Loop
    while True:
        try:
            query = console.input("\n[bold magenta]Ask Dilli Darshan (or 'history' / 'new' / 'exit' / 'stats' / '/agent <task>'): [/bold magenta]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting Dilli Darshan. Safe travels and have a wonderful journey![/bold yellow]\n")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            console.print("\n[bold yellow]Thank you for exploring with Dilli Darshan! Have a wonderful journey across Delhi.[/bold yellow]\n")
            break

        if query.lower() in ("history", "/history"):
            show_session_history(chat_session)
            continue

        if query.lower() in ("new", "/new", "resetchat", "/resetchat"):
            chat_session.reset_session()
            console.print("\n[bold green]✓ Chat session memory reset. Starting a fresh conversation![/bold green]\n")
            continue

        if query.lower() == "stats":
            show_stats()
            continue

        if query.lower().startswith(("/agent", "agent ")):
            goal = re.sub(r"^/?agent\s*", "", query, flags=re.IGNORECASE).strip()
            if goal:
                run_agent_workflow(goal)
            else:
                console.print("[yellow]Usage: /agent <goal or place to inspect live>[/yellow]")
            continue

        if query.lower() in ("clear", "reset"):
            confirm = console.input("[bold red]Are you sure you want to clear ChromaDB? (y/N): [/bold red]").strip().lower()
            if confirm in ("y", "yes"):
                embedding.clear_database()
            continue

        run_query_pipeline(query, chat_session=chat_session, auto_cache_enabled=auto_cache_flag)


if __name__ == "__main__":
    main()
