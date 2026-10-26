# How I Tracked User Preferences Without Cookies Using Hindsight

Most web architectures treat user preference tracking as a choice between two bad options: stuffing serialized state into client-side cookies or maintaining rigid columns in a relational database. When we built DilliDarshan—an autonomous travel companion and retrieval engine for navigating New Delhi—neither approach survived first contact with real users.

Travelers don't announce constraints in onboarding forms. They drop critical details mid-conversation: *"I have a bad knee so I can't climb steep stairs,"* or *"I'm strictly vegetarian—where should I eat after visiting the mosque?"* Storing these nuances in browser cookies means fighting 4KB limits, managing intrusive consent banners, and losing context across devices. Storing them in database tables requires custom entity extractors and schema migrations for every new preference type.

To solve this, we decoupled preference tracking from the browser entirely. By integrating Vectorize's memory engine, we built a cookie-free personalization pipeline that extracts, retains, and recalls implicit human constraints across multi-turn sessions using dedicated agent memory banks.

## What the System Does and How It Hangs Together

DilliDarshan is a domain-specialized conversational retrieval engine for Delhi tourism. Standard vector search across city landmarks breaks down quickly: monuments have strict ticket windows, metro lines have dozens of numbered gates, and real-time conditions require live verification.

We structured the backend around a hybrid retrieval-augmented generation (RAG) architecture:

1. **Verified Domain Store (`ChromaDB`)**: Pre-indexes hundreds of semantic chunks across 15+ major Delhi monuments with contextual breadcrumbs to prevent vector drift.
2. **Live Web Fallback (`TinyFish SDK`)**: Scrapes live Markdown from authoritative sources when users ask about real-time events or unindexed spots.
3. **Generation Engine (`Google Gemini 3.8 Flash`)**: Synthesizes verified chunks into direct, local-insider guidance with zero conversational throat-clearing.
4. **Agent Memory Bank (`Vectorize Hindsight Cloud`)**: Serves as the persistent store for traveler identities, mobility limitations, dietary rules, and dialogue history.

Because the front end is a single-page companion app running across browsers and mobile webviews, we eliminated session cookies and local storage sync entirely. User-specific state lives exclusively server-side inside an isolated memory bank.

## The Core Technical Story: Why Cookies and Relational Tables Failed Us

In an LLM-driven application, tracking user preferences is fundamentally different from storing a cart ID or a dark-mode toggle.

### Cookies Break on Unstructured Semantic State

Cookies were designed for small, discrete tokens. Using them for conversational context hits immediate walls:

- **Size and Network Overhead**: Browser cookies max out at 4,096 bytes per domain. Serializing conversational context and visited landmarks quickly breaches that limit, bloating HTTP request headers.
- **Cross-Device Context Collapse**: A tourist researches on a laptop by day and navigates on their phone by evening. Cookie-based state isolates those sessions completely unless you build a backend synchronization service.
- **Privacy and Ephemeral Lifetimes**: Tracking cookies require invasive consent banners and trigger browser cleanups (like Safari's ITP), which wipe client storage after days of inactivity.

### Relational Schemas Cannot Keep Pace with Conversational Nuance

Our first draft considered an internal PostgreSQL table with columns like `is_vegetarian` and `has_mobility_impairment`.

This pattern collapses in production. What happens when a user remarks: *"My mother gets tired after thirty minutes of walking"*? Or *"I avoid crowded indoor bazaars because of dust allergies"*? 

To capture that in SQL, you either write brittle regex/NER parsers to force sentences into rigid columns, or you dump arbitrary JSON into a `jsonb` column. If you dump it into `jsonb`, you lose the ability to query that context semantically when assembling generation prompts. You cannot easily ask Postgres: *"Retrieve only past health constraints relevant to climbing the minaret at Jama Masjid."*

### The Agent Memory Alternative

We needed an engine that could accept raw dialogue turns, extract semantic preferences autonomously, and expose a low-latency vector recall interface tailored for LLM prompts. 

Rather than maintaining custom pgvector clustering, embedding recalculations, and entity graph pipelines, we integrated the [open-source Hindsight engine](https://github.com/vectorize-io/hindsight). Under this model, each tourist is allocated an isolated memory bank on the cloud service. 

We interact with memory through two fundamental primitives:
- **Recall**: Given the user's current query, search their memory bank for semantically relevant preferences, physical constraints, and visited monuments.
- **Retain**: After answering, asynchronously stream the conversation turn to the memory bank so the engine can extract newly stated facts in the background.

This architectural shift allowed us to eliminate client-side state entirely while giving the assistant an enduring, cross-device memory. For a deeper breakdown of this paradigm, see Vectorize's guide on [agent memory architecture](https://vectorize.io/what-is-agent-memory).

## Code-Backed Explanations: Implementing Zero-Cookie Memory

Here is how we implemented the zero-cookie architecture in Python using FastAPI, ChromaDB, and `hindsight-client`.

### 1. Deterministic Identity and Thread-Safe Bank Provisioning

Rather than issuing a session cookie, we derive a deterministic integer `user_id` and unique `bank_id` from the user's verified identity (such as an email address). The client passes this ID in standard JSON payloads.

In `memory.py`, we initialize the client using thread-local storage to avoid cross-thread event loop collisions in FastAPI worker pools:

```python
import os, zlib, threading
from typing import Optional
from hindsight_client import Hindsight

_thread_local = threading.local()
hindsight_api_key = os.getenv("HINDSIGHT_API_KEY", "").strip()
hindsight_base_url = os.getenv("HINDSIGHT_API_URL", "https://api.hindsight.vectorize.io")

def get_hindsight_client() -> Optional[Hindsight]:
    if not hindsight_api_key:
        return None
    if not hasattr(_thread_local, "client") or _thread_local.client is None:
        _thread_local.client = Hindsight(base_url=hindsight_base_url, api_key=hindsight_api_key)
    return _thread_local.client

def get_user_id_from_email(email: str) -> int:
    clean_email = email.strip().lower() if email else "tourist@delhi.com"
    return abs(zlib.crc32(clean_email.encode("utf-8"))) % 100000000 + 1

def ensure_bank_exists(bank_id: str, bank_name: str = "Dilli Darshan Guest") -> None:
    h_client = get_hindsight_client()
    if not h_client:
        return
    try:
        h_client.create_bank(
            bank_id=bank_id,
            name=bank_name,
            mission="Track identity, travel style, dietary restrictions, budget, and accessibility needs.",
            disposition={"skepticism": 2, "literalism": 4, "empathy": 5}
        )
    except Exception:
        pass
```

By setting `empathy: 5` and `skepticism: 2`, we configure the bank to prioritize human physical constraints and personal preferences rather than treating user statements as factual claims requiring external corroboration.

### 2. Contextual Recall and Grounded Prompt Assembly

When a user submits a question, we execute a targeted memory recall before querying our vector database or invoking the LLM. 

In `retrieval.py`, `DilliDarshanChatSession.send_message` queries the memory bank using a focused semantic query, then formats recalled facts into an authoritative instruction block:

```python
def send_message(self, raw_query: str, user_id: int | None = None, user_name: str = "Tourist", **kwargs):
    # 1. Recall user constraints and implicit preferences from Hindsight
    recalled = memory.recall_user_memories(user_id=user_id, query=raw_query, user_name=user_name)
    formatted_memory_context = recalled.get("formatted_memory_context", "")

    # 2. Resolve contextual place inheritance ('it' -> 'Jama Masjid')
    effective_query, active_places, is_followup = self.resolve_contextual_query(
        raw_query, recalled_memories=recalled.get("memories", [])
    )

    # 3. Retrieve verified local guidebook chunks from ChromaDB
    chunks, status, _ = retrieve_from_chromadb(effective_query, target_places=active_places)

    # 4. Assemble grounded generation prompt with injected constraints
    memory_prompt_block = (
        "Hindsight Cloud Agent Memory (User Profile & Constraints):\n"
        f"{formatted_memory_context}\n\n"
        "Strictly respect all user constraints, accessibility needs (e.g., knee injuries, "
        "stairs avoidance), and dietary restrictions (e.g., vegetarian) in your answer!"
    )

    grounded_turn_prompt = (
        f"Verified Local Guidebook Context:\n{format_chunks(chunks)}\n\n"
        f"{memory_prompt_block}\n\n"
        f"User Question: {raw_query}\nActive Landmark: {', '.join(active_places)}\n\n"
        "Provide an accurate, insider response. Strictly obey the zero-greeting directive:"
    )

    response, model_used = execute_gemini_generation(grounded_turn_prompt)
```

Instead of bloating prompts with raw chat transcripts that exhaust token budgets and distract the model, Hindsight returns only the distilled facts relevant to the active question.

### 3. Non-Blocking Asynchronous Retention

Waiting for a memory service to process and commit facts before sending an answer adds 400ms to 800ms of lag to every turn.

To eliminate this overhead, we use Hindsight's asynchronous retention mode (`retain_async=True`). As detailed in the [Hindsight documentation](https://hindsight.vectorize.io/), this acknowledges the write immediately over HTTP while the extraction pipeline runs in the background:

```python
def retain_chat_turn(user_id: Optional[int], user_name: str, user_query: str, bot_answer: str, active_places: list[str]) -> bool:
    bank_id = get_bank_id_for_user(user_id, user_name)
    ensure_bank_exists(bank_id, f"Dilli Darshan - {user_name}")

    turn_content = (
        f"Tourist ({user_name}) said: \"{user_query}\"\n"
        f"Active Landmark discussed: {', '.join(active_places)}\n"
        f"Guide replied: \"{bot_answer[:600].replace(chr(10), ' ')}\""
    )

    h_client = get_hindsight_client()
    if h_client:
        try:
            h_client.retain(
                bank_id=bank_id,
                content=turn_content,
                context="conversation_turn_and_preferences",
                metadata={"source": "chat_turn", "active_places": ",".join(active_places)},
                retain_async=True
            )
            return True
        except Exception as e:
            logger.warning(f"Hindsight retention failed: {e}")
    return False
```

The user receives their response immediately, while Hindsight extracts newly stated facts in the background and indexes them into the user's bank.

## Results and Behavior: A Real Multi-Turn Interaction

Consider an actual interaction trace from our integration test suite (`verify_backend.py`). A traveler named Aarav interacts with DilliDarshan across multiple turns without any client cookies active.

### Turn 1: Implicit Constraint Disclosure

Aarav registers with his name and email, creating bank `dilli_darshan_tourist_71829341`. He sends his first query:

> **User**: *"Hi, I am strictly vegetarian and I have a knee injury so I cannot climb steep stairs. Tell me about Jama Masjid."*

The pipeline executes:
1. **Recall**: Returns his registration profile (`Heritage & History`).
2. **Place Resolution**: Detects `jama_masjid` as the active landmark.
3. **Retrieval**: Fetches verified chunks for Jama Masjid from ChromaDB.
4. **Generation**: Prompted with Aarav's knee injury, Gemini highlights entrance accessibility:

> **DilliDarshan**:
> *Jama Masjid sits on an elevated sandstone plinth 30 feet above street level. While the Eastern Gate features 35 steep red sandstone steps, visitors with knee injuries should use Gate 2 (Northern Gate) or Gate 1 (Southern Gate), which offer gentler inclines and entrance ramps. Avoid ascending the narrow 130-step spiral staircase of the southern minaret.*

5. **Retention**: The turn is committed asynchronously. Hindsight extracts two core facts into Aarav's bank:
   - *Physical constraint: Knee injury, avoid steep steps and minaret climbs.*
   - *Dietary restriction: Strictly vegetarian.*

### Turn 2: The Follow-Up Test

A minute later, Aarav submits a follow-up query containing zero explicit mentions of the monument, his injury, or his diet:

> **User**: *"What are the timings there and where should I eat nearby?"*

A traditional stateless RAG bot fails here: it doesn't know what *"there"* refers to, and it recommends Old Delhi's most famous nearby eateries—legendary non-vegetarian kebab houses (Karim's and Al Jawahar).

Here is how our pipeline handles it:
1. **Hindsight Recall**: Recalls prior context (knee injury, vegetarian diet, recent landmark: Jama Masjid).
2. **Contextual Query Resolution**: Detects pronoun *"there"*, rewriting the effective search query to `Jama Masjid: What are the timings and nearby food spots?`.
3. **ChromaDB Vector Retrieval**: Pulls verified timing chunks for Jama Masjid and culinary chunks for Old Delhi.
4. **Generation with Constraint Injection**: The LLM synthesizes the answer, strictly respecting the vegetarian constraint stored in Hindsight:

> **DilliDarshan**:
> *Jama Masjid is open to tourists daily from 7:00 AM to 12:00 PM and 1:30 PM to 6:30 PM (closed to non-Muslim visitors during afternoon namaz). For dining near Gate 1 that adheres to strict vegetarian dietary standards, bypass the nearby Mughlai meat stalls along Matia Mahal and proceed 300 meters into Dariba Kalan and Paranthe Wali Gali. Recommended spots include:*
> 
> *1. **Pandit Gaya Prasad Shiv Charan (Est. 1872)**: Famous pure-vegetarian tawa-fried stuffed paranthas served with pumpkin curry and mint chutney.*
> *2. **Kake Di Hatti (Church Mission Road)**: Iconic for tandoori naans and pure-vegetarian dal makhani.*
> *3. **Jalebi Wala (Dariba Corner)**: Traditional dessert fried in pure desi ghee.*

Without setting a single cookie, without asking Aarav to repeat his constraints, and without running a database migration, the system adapted its route advice and restaurant recommendations perfectly to his context.

## Lessons Learned: Key Takeaways for Agent Memory

Building a persistent, cookie-free conversational application surfaced four practical lessons for shipping agent memory:

### 1. Decouple Domain Knowledge from Human Context
Domain knowledge (monument operating hours, ticket costs, architectural history) is static and shared across all users. Personal preferences (knee injuries, food allergies, budget limits) are volatile and user-specific. Mixing them in the same vector index creates permission boundaries and index pollution. Keeping domain knowledge in local ChromaDB collections and traveler memory in partitioned Hindsight Cloud banks cleanly separated our concerns.

### 2. Never Block the Response on Memory Retention
Your prompt assembly pipeline must read memory synchronously to produce an accurate answer, but it should **never** write memory synchronously. Memory extraction is computationally heavy. By delegating writes to `h_client.retain(..., retain_async=True)`, we kept end-to-end API latency under 900ms on `gemini-3.8-flash`. If retention fails due to a network glitch, the user still receives their answer without delay.

### 3. Contextual Query Rewriting Is Essential for RAG
Even the best memory engine cannot fix a bad vector search. If a user asks *"Where should I eat there?"*, vector search against a monument database returns low-confidence noise. You must inspect the query for follow-up pronouns (`it`, `there`), check active memory state for the current subject, and rewrite the query (`Jama Masjid: Where should I eat?`) before executing retrieval. Memory provides context; query rewriting makes that context actionable for vector search.

### 4. Memory Banks Need Explicit Missions and Dispositions
A memory engine is not a passive key-value cache. In Hindsight, setting the bank's `mission` and `disposition` parameters was the difference between useful personalization and noisy hallucinations. By defining the mission specifically around travel planning (diet, budget, mobility, visited spots) and tuning disposition toward empathy rather than strict literalism, the extraction pipeline learned to ignore irrelevant small talk while aggressively capturing physical and lifestyle constraints that affect travel.

## Conclusion

The web has outgrown cookies, and conversational AI has outgrown rigid relational user tables. When your application's primary interface is open-ended natural language, human preferences will always spill outside predefined form fields.

By pairing a verified local RAG pipeline with dedicated cloud memory banks, we built a companion that remembers who you are, how you travel, and what physical limitations you have—without ever setting a cookie in your browser or running a database migration. For any team building agentic workflows where human context matters across sessions, treating memory as a dedicated architectural service is the cleanest way forward.
