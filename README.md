# 🏛️ DilliDarshan
### *Next-Generation RAG & Autonomous Live Tour Companion for New Delhi*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_RAG-orange.svg?logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.8_Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![TinyFish](https://img.shields.io/badge/TinyFish-Live_Web_Engine-00C49F.svg)](https://tinyfish.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](#)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Repository File Structure](#-repository-file-structure)
- [Knowledge Base & Pre-Indexed Monuments](#-knowledge-base--pre-indexed-monuments)
- [RAG & Intelligence Engine Deep Dive](#-rag--intelligence-engine-deep-dive)
  - [1. Semantic Markdown Chunking & Context Injection](#1-semantic-markdown-chunking--context-injection-chunkingpy)
  - [2. Resilient Embedding & Triple-Key Failover](#2-resilient-embedding--triple-key-failover-embeddingpy)
  - [3. Two-Tier Retrieval & Corrective RAG Decision Gate](#3-two-tier-retrieval--corrective-rag-decision-gate-retrievalpy)
  - [4. TinyFish Live Web Engine & Autonomous Browser Agent](#4-tinyfish-live-web-engine--autonomous-browser-agent)
  - [5. Self-Learning Auto-Caching Engine](#5-self-learning-auto-caching-engine)
  - [6. Multi-Turn Conversational Memory & Place Inheritance](#6-multi-turn-conversational-memory--place-inheritance)
  - [7. Personality, Travel Styles & Zero-Greeting Directness](#7-personality-travel-styles--zero-greeting-directness)
- [Database Schema (SQLite Persistence)](#-database-schema-sqlite-persistence)
- [FastAPI REST API Reference](#-fastapi-rest-api-reference)
- [Frontend Companion UI](#-frontend-companion-ui)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
  - [1. Starting the FastAPI Server & Web Companion](#1-starting-the-fastapi-server--web-companion)
  - [2. Interactive CLI Mode](#2-interactive-terminal-cli-mode)
  - [3. Ingestion & Database Management](#3-ingestion--database-management)
  - [4. Automated Backend & Frontend Verification Suite](#4-automated-backend--frontend-verification-suite)
- [Environment Variables](#-environment-variables)
- [Tech Stack](#-tech-stack)
- [License](#-license)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Overview

**DilliDarshan AI** is an enterprise-grade, domain-specialized conversational AI and tourism intelligence platform built specifically for the National Capital Territory of Delhi, India.

Traditional travel bots either hallucinate outdated monument timings and ticket prices or provide generic Wikipedia summaries. **DilliDarshan AI** solves this with a **Hybrid Corrective RAG (Retrieval-Augmented Generation)** architecture:

1. **Local Domain Grounding**: Searches a high-precision vector database (`ChromaDB`) holding verified guidebooks across 15+ major Delhi monuments and museums.
2. **Autonomous Web Fallback & Live Research**: Integrates the **TinyFish Python SDK** to autonomously search the web and scrape live Markdown when real-time events (weather, crowd, metro closures, current exhibitions) or unknown landmarks are queried.
3. **Self-Learning Auto-Cache**: Automatically structures web discoveries into standardized 5-section guidebooks and caches them into `ChromaDB` for instant zero-latency retrieval in subsequent queries.
4. **Stateful Conversational Memory**: Understands contextual follow-ups ("*Which metro gate is closest to reach there?*") by dynamically inheriting active landmarks.
5. **Personalization & SQLite Persistence**: Customizes tone, recommendations, and language (English / Hindi) based on user travel personas (*Heritage & History*, *Food & Bazaars*, *Budget & Transit*, *Family Leisure*).

---

## 🏛 System Architecture

The following diagram illustrates the complete end-to-end data pipeline from user query to verified generation:

```
                              ┌────────────────────────────────────────┐
                              │  User Interaction (Web UI / Terminal)   │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                               ┌───────────────────────────────────────┐
                               │   DilliDarshanChatSession (Stateful)  │
                               │   - Contextual Place Resolution       │
                               │   - Pronoun & Follow-up Rewriting     │
                               └──────────────────┬────────────────────┘
                                                  │
                                                  ▼
                               ┌───────────────────────────────────────┐
                               │     Stage 1: Zero-Latency Router      │
                               │  Classifies into Category Metadata    │
                               │  (History, Tickets, Metro, Food, etc.) │
                               └──────────────────┬────────────────────┘
                                                  │
                                                  ▼
                               ┌───────────────────────────────────────┐
                               │ Stage 2: Two-Tier ChromaDB Retrieval  │
                               │ Tier 1: Category-Filtered Vector Scan │
                               │ Tier 2: Broad Vector Fallback         │
                               └──────────────────┬────────────────────┘
                                                  │
                                                  ▼
                               ┌───────────────────────────────────────┐
                               │     Corrective RAG Decision Gate      │
                               │   - Similarity Check (Dist <= 0.48)   │
                               │   - Fuzzy Landmark Verification       │
                               │   - Live Query Keyword Detector       │
                               └───────┬──────────────────────┬────────┘
                                       │                      │
                   [LOCAL_HIT]         │                      │ [WEB_FALLBACK / HYBRID]
                                       ▼                      ▼
                    ┌─────────────────────────┐   ┌───────────────────────────────┐
                    │ ChromaDB Vector Chunks  │   │ TinyFish Live Web Engine      │
                    │ (Verified Local Guides) │   │ - Live Web Search (tf.search) │
                    └────────────┬────────────┘   │ - Clean MD Scraping (fetch)   │
                                 │                │ - Autonomous Browser Agent    │
                                 │                └───────────────┬───────────────┘
                                 │                                │
                                 └───────────────┬────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────────┐
                               │   Stage 3: Grounded Gemini LLM        │
                               │   - System Instruction: Local Guide   │
                               │   - Profile & Style Prompt Injection  │
                               │   - Triple-Key Resilient Client       │
                               │   - Model Cascading (3.8 -> 3.6 -> 2.5)│
                               └──────────────────┬────────────────────┘
                                                  │
                                                  ▼
                               ┌───────────────────────────────────────┐
                               │ Post-Processing & Formatting Engine   │
                               │ - Zero-Greeting Enforcement           │
                               │ - Markdown -> Styled Semantic HTML    │
                               │ - Source Attribution Badge Extraction │
                               └───────┬──────────────────────┬────────┘
                                       │                      │
                                       ▼                      ▼
                      ┌─────────────────────────┐   ┌───────────────────────────┐
                      │  SQLite users.db Audit  │   │ Self-Learning Auto-Cache  │
                      │  - Saves chat_logs      │   │ - Synthesizes 5-sec Guide │
                      │  - Updates profile      │   │ - Ingests new data/*.md   │
                      └─────────────────────────┘   └───────────────────────────┘
```

---

## ✨ Key Features

- **🛡️ Triple-Key API Resilience (`ResilientGeminiClient`)**:
  Eliminates downtime caused by API quota exhaustion (HTTP 429) or transient network hiccups by rotating between `API_Key`, `API_Key_B1`, and `API_Key_B2` automatically.
- **⚡ LLM Model Cascading**:
  Executes generation using `gemini-3.8-flash` by default; automatically cascades down to `gemini-3.6-flash`, `gemini-2.5-flash`, and `gemini-flash-latest` if rate limits or outages occur.
- **🎯 Corrective RAG Decision Gate**:
  Ensures the bot never hallucinates across monuments. Validates that retrieved chunks belong to the requested landmark via fuzzy ratio matching ($\ge 75\%$).
- **🌐 Real-Time Live Web Ingestion (TinyFish SDK)**:
  Queries the live web using TinyFish's high-speed search and scraping engine to obtain up-to-the-minute updates on Delhi events, weather, ticket hikes, or temporary closures.
- **🤖 Autonomous Browser Agent**:
  Includes `tinyfish_agent_live_check`, deploying an AI browser agent to navigate official government and tourism portals directly.
- **🧠 Self-Learning & Auto-Ingestion**:
  When a user asks about a Delhi location not yet in the knowledge base, DilliDarshan AI researches it on the fly, synthesizes a complete 5-section guidebook, chunks it, and saves it into `ChromaDB`.
- **🗣️ Multi-Turn Conversational Memory**:
  Maintains dialogue context across turns. If you ask "*Tell me about Lal Qila*", followed by "*Which gate has the closest metro exit?*", it recognizes that "*there*" refers to Lal Qila without needing prompt restatements.
- **🚫 Zero-Greeting Directness**:
  Enforces a strict prompt rule and regex safety net to ban repetitive pleasantries ("*Hello*", "*Namaste*", "*Hi Saif*"). The bot provides punchy, factual, high-value advice from the very first word.
- **🎨 Personalized Travel Styles**:
  Customizes output based on four distinct travel personas:
  - **Heritage & History**: Emphasizes dynasties, Mughal/Sultanate architecture, and historical significance.
  - **Street Food & Bazaars**: Highlights iconic culinary spots, kebabs, chaat, sweets, and narrow bazaar trails.
  - **Budget & Transit / Metro**: Focuses on Delhi Metro stations, lines, exit gates, fares, and economical hacks.
  - **Family Leisure & Gardens**: Focuses on accessibility, seating, lush lawns, stroller friendliness, and relaxed strolls.
- **🌐 Bilingual Output (English & Hindi)**:
  Supports high-quality, fluent Hindi (Devanagari script) with accurate bilingual landmarks and metro signage.
- **📱 Single-Page Companion UI**:
  Includes an interactive, responsive web frontend with ambient slideshows, live chat bubbles, quick prompt chips, and a profile customization drawer.

---

## 📂 Repository File Structure

```text
d:\Dilli Darshan backend/
│
├── api.py                  # FastAPI application exposing REST endpoints & serving companion UI
├── chunking.py             # Semantic Markdown splitting, category detection & context header injection
├── embedding.py            # ChromaDB client, Gemini Embeddings, Resilient Key Manager & TinyFish client
├── ingesting.py            # Batch document ingestion pipeline & autonomous CLI place discovery
├── retrieval.py            # Core RAG engine: Two-Tier retrieval, Decision Gate, TinyFish & ChatSession
├── database.py             # SQLite persistence: user profiles and multi-turn chat audit logs
├── verify_backend.py       # Comprehensive automated integration test suite
│
├── frontend.html           # Full-featured Single-Page Application (SPA) companion frontend
├── requirements.txt        # Python dependency declarations
├── places.txt              # Summary catalog of pre-indexed Delhi monuments and chunk distributions
├── LICENSE                 # MIT License (Copyright 2026 MD SAIF JAWED)
│
├── assets/                 # Monument visual assets for frontend backdrop slideshow
│   ├── india-gate.jpg
│   ├── lotus-temple.jpg
│   ├── qutub-minar.jpg
│   └── red-fort.jpg
│
├── chroma_db/              # Persistent ChromaDB vector database directory
│   └── chroma.sqlite3      # Vector storage, embeddings & HNSW cosine index
│
├── data/                   # Raw documents, monument PDFs, and auto-cached Markdown guides
│   ├── Cannaught place.pdf
│   └── Humayun's Tomb.pdf
│
└── users.db                # SQLite database storing users, preferences, and chat interaction logs
```

---

## 🏛 Knowledge Base & Pre-Indexed Monuments

The default vector database contains **504 high-density semantic chunks** extracted from verified travel guides across **15 iconic New Delhi landmarks**:

| # | Monument / Landmark | Place ID | Total Chunks | Primary Categories Covered |
|---|---------------------|----------|:------------:|----------------------------|
| 1 | **National Crafts Museum** | `craft_museum` | 59 | History, Exhibits, Food, Transit |
| 2 | **Lal Qila (Red Fort)** | `lal_qila` | 53 | History, Architecture, Light & Sound, Metro |
| 3 | **National Rail Museum** | `rail_museum` | 51 | Exhibits, Toy Train, Timings, Transit |
| 4 | **Air Force Museum** | `air_force_museum` | 46 | Vintage Aircraft, Hangar, Timings, Metro |
| 5 | **Pradhanmantri Sangrahalaya** | `pm_snagyaly` | 46 | Modern History, Interactive Galleries, Tickets |
| 6 | **Jama Masjid** | `jama_masjid_guide` | 44 | Mughal Architecture, Courtyard, Old Delhi Food |
| 7 | **Qutub Minar Complex** | `qutub_minar` | 44 | Iron Pillar, Sultanate History, Tickets, Metro |
| 8 | **Akshardham Temple** | `akshardham_guide` | 40 | Exhibition, Water Show, Etiquette, Security |
| 9 | **Gandhi Smriti Museum** | `gandhi_samriti_museum` | 33 | Birla House, Martyr's Column, Timings |
| 10 | **Indira Gandhi Memorial** | `indra_gandhi_museum` | 33 | Historical Residence, Memorial, Free Entry |
| 11 | **Connaught Place (CP)** | `cannaught_place` | 12 | Architecture, Inner/Outer Circles, Eateries |
| 12 | **Humayun's Tomb** | `humayun's_tomb` | 12 | Charbagh Garden, Persian Architecture, Tickets |
| 13 | **India Gate & Kartavya Path** | `india_gate` | 11 | War Memorial, Evening Stroll, Street Snacks |
| 14 | **Hauz Khas Complex** | `hauz_khas` | 10 | Lake, Fort Ruins, Village Cafes, Deer Park |
| 15 | **Jantar Mantar** | `jantar_mantar` | 10 | Astronomical Instruments, Samrat Yantra |

---

## 🔍 RAG & Intelligence Engine Deep Dive

### 1. Semantic Markdown Chunking & Context Injection (`chunking.py`)

Standard text chunkers break text arbitrarily by character count, resulting in disconnected paragraphs without context. `chunking.py` implements a structured hierarchical chunking pipeline:

1. **Markdown Header Splitting**: Uses `MarkdownHeaderTextSplitter` along `#` (monument), `##` (section), and `###` (subsection).
2. **Context Header Injection**: Prepends a contextual breadcrumb to every chunk before embedding:
   ```text
   [Place: Lal Qila | Section: Timings & Tickets > Sound & Light Show]
   The evening sound and light show runs in Hindi and English...
   ```
   *Benefit*: Guarantees that chunks never lose their monument association during cosine similarity matching, even when sub-split.
3. **Category Tagging (`detect_category`)**: Automatically classifies chunks into semantic tags:
   - `history_and_architecture`
   - `timings_and_tickets`
   - `transit_and_metro`
   - `food_and_dining`
   - `quick_facts`
   - `general`

### 2. Resilient Embedding & Triple-Key Failover (`embedding.py`)

- **Model**: Google `gemini-embedding-2`
- **Differentiated Task Types**:
  - `RETRIEVAL_DOCUMENT` used when indexing chunks into `ChromaDB`.
  - `RETRIEVAL_QUERY` used when generating query search vectors.
- **`ResilientGeminiClient`**:
  Wraps the Google GenAI SDK. If an API key hits rate limits (429 Quota Exceeded) or network errors, it instantly switches to `API_Key_B1` or `API_Key_B2` without failing the user request:

```python
# Automatic Failover Flow
Primary: API_Key ──[429 Quota Exceeded]──► Backup 1: API_Key_B1 ──► Backup 2: API_Key_B2
```

### 3. Two-Tier Retrieval & Corrective RAG Decision Gate (`retrieval.py`)

When a query arrives, `retrieval.py` executes a multi-stage evaluation:

1. **Zero-Latency Category Router**:
   Inspects query keywords to determine relevant categories (e.g., "*metro*" maps to `transit_and_metro` + `quick_facts`).
2. **Two-Tier Search**:
   - **Tier 1 (Filtered)**: Vector search strictly within matching categories.
   - **Tier 2 (Broad)**: Unfiltered vector search across the entire collection.
3. **Corrective RAG Decision Gate**:
   - Computes cosine distance ($Similarity = 1.0 - Distance$).
   - Filters out candidates with $Distance > 0.48$ ($Similarity < 0.52$).
   - **Landmark Consistency Check**: Validates that retrieved metadata matches the user's targeted place using substring and fuzzy token matching (`difflib.SequenceMatcher` $\ge 0.75$).
4. **Routing Status Output**:
   - `LOCAL_HIT`: Valid local chunks found with high similarity.
   - `HYBRID_SUPPLEMENT`: Local chunks found, but user asked for live/current details ("*today*", "*current crowd*", "*2026 exhibitions*").
   - `WEB_FALLBACK`: Landmark is not present in local database; triggers TinyFish web research.
   - `EMPTY_DB`: Vector database is empty; triggers web fallback.

### 4. TinyFish Live Web Engine & Autonomous Browser Agent

When a live event or new landmark is requested:
1. **Live Web Search & Scrape (`tinyfish_live_search_and_fetch`)**:
   Uses `tf_client.search.query(location="New Delhi, India")` to find real-time sources, filters out social media noise, and calls `tf_client.fetch.get_contents(format="markdown")` to extract clean text.
2. **Autonomous Browser Agent (`tinyfish_agent_live_check`)**:
   Spawns `tf_client.agent.run` to navigate live government or ticketing portals, confirming ticket availability, operational hours, and visitor notices.

### 5. Self-Learning Auto-Caching Engine

Whenever `WEB_FALLBACK` triggers for an unrecognized Delhi landmark:
1. TinyFish fetches live web pages.
2. Gemini synthesizes a comprehensive 5-section guidebook:
   - `## 1. History & Architecture`
   - `## 2. Timings & Tickets`
   - `## 3. How to Reach & Metro`
   - `## 4. Nearby Iconic Food Spots`
   - `## 5. Quick Reference & Facts`
3. The file is saved as `data/<place_id>.md`.
4. It is immediately chunked via `chunking.py` and embedded into `ChromaDB`.
5. *Result*: The system dynamically learns and expands its local knowledge base autonomously!

### 6. Multi-Turn Conversational Memory & Place Inheritance

`DilliDarshanChatSession` manages conversation history:
- Tracks `last_active_places`.
- Resolves pronoun references ("*it*", "*there*", "*this monument*") and standalone category queries ("*What is the ticket fee?*").
- Rewrites ambiguous follow-up queries to explicit search phrases (e.g. `Lal Qila: What is the ticket fee?`) for vector retrieval while maintaining natural conversational dialogue.

### 7. Personality, Travel Styles & Zero-Greeting Directness

- **Secular & Culturally Immersed**: Operates as a seasoned local insider rather than a dry encyclopedia.
- **Zero Greetings**: Strictly forbids "*Hello*", "*Namaste*", or conversational throat-clearing. Jumps straight into practical facts and transit recommendations.
- **Language Adaptability**: Responds in natural English or expressive Hindi (Devanagari) with bilingual metro/gate notations.

---

## 🗄 Database Schema (SQLite Persistence)

All user accounts, preferences, and interaction histories are persisted in `users.db` using WAL mode (`PRAGMA journal_mode = WAL;`) for high concurrency:

```sql
-- 1. Tourist Profiles
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    travel_style TEXT DEFAULT 'Heritage & History',
    language TEXT DEFAULT 'English',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for case-insensitive email lookups
CREATE INDEX IF NOT EXISTS idx_users_email_lower ON users(LOWER(email));

-- 2. Audit Trail & Chat History
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    session_key TEXT,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT,
    routing_status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔌 FastAPI REST API Reference

The backend exposes a clean REST API built with FastAPI:

### 1. Save or Update User Profile
- **Endpoint**: `POST /user/profile`
- **Description**: Creates or updates a tourist profile in SQLite.

**Request Payload:**
```json
{
  "name": "Saif Jawed",
  "email": "saif@example.com",
  "travel_style": "Heritage & History",
  "language": "English"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "user_id": 1,
  "name": "Saif Jawed",
  "email": "saif@example.com",
  "travel_style": "Heritage & History",
  "language": "English"
}
```

---

### 2. Conversational Chat
- **Endpoint**: `POST /chat`
- **Description**: Submits a query to the stateful RAG pipeline.

**Request Payload:**
```json
{
  "query": "What are the timings and entry fee for Red Fort?",
  "user_id": 1,
  "user_name": "Saif Jawed",
  "travel_style": "Heritage & History",
  "language": "English",
  "session_id": "user_1"
}
```

**Response (200 OK):**
```json
{
  "answer": "<div><strong>Red Fort (Lal Qila)</strong> is open from <strong>9:30 AM to 4:30 PM</strong>, Tuesday through Sunday (closed on Mondays)...</div>",
  "raw_markdown": "**Red Fort (Lal Qila)** is open from **9:30 AM to 4:30 PM**...",
  "source": "Lal Qila Archives (ChromaDB)",
  "routing_status": "LOCAL_HIT",
  "active_places": ["lal_qila"],
  "turn_number": 1,
  "session_id": "user_1"
}
```

---

### 3. Reset Session Memory
- **Endpoint**: `POST /chat/reset`
- **Description**: Clears conversation memory and active landmark context for a user or session.

**Request Payload:**
```json
{
  "user_id": 1
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Session 'user_1' successfully reset"
}
```

---

### 4. System Health Check
- **Endpoint**: `GET /health` or `GET /api/health`
- **Description**: Returns live diagnostics of vector collection, active LLM model, user count, and logged interactions.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "DilliDarshan AI Backend",
  "chromadb_chunks": 504,
  "active_model": "gemini-3.8-flash",
  "gemini_key_active": true,
  "registered_users": 1,
  "logged_chat_turns": 4,
  "active_sessions": 1
}
```

---

### 5. Fetch Profile by User ID
- **Endpoint**: `GET /user/profile/{user_id}`
- **Response (200 OK):**
```json
{
  "user_id": 1,
  "name": "Saif Jawed",
  "email": "saif@example.com",
  "travel_style": "Heritage & History",
  "language": "English",
  "created_at": "2026-10-18 17:30:00",
  "updated_at": "2026-10-18 17:30:00"
}
```

---

## 💻 Frontend Companion UI

The included `frontend.html` provides a modern Single-Page Application (SPA) designed with pure vanilla CSS and HTML5:

1. **Ambient Monument Slideshow**: Full-screen cinematic backgrounds transitioning smoothly between Delhi's iconic monuments (*Red Fort, India Gate, Lotus Temple, Qutub Minar*).
2. **Tourist Profile Drawer**: Instant configuration of traveler persona (*Heritage & History*, *Food & Bazaars*, *Budget & Metro*, *Family Leisure*) and language preference (*English / Hindi*).
3. **Conversational Chat Canvas**: Renders formatted HTML responses, typing indicators, quick prompt chips, and source attribution pills (`.source-badge`).
4. **Resilient Network Handling**: Automatically recovers from connectivity issues with client-side fallback suggestions.

---

## 🚀 Installation & Setup

### Prerequisites
- Python **3.10** or higher
- A Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/))
- *(Optional)* A TinyFish API key (from [TinyFish](https://tinyfish.io/)) for live web research and autonomous browser agents

### Step 1: Clone the Repository
```bash
git clone https://github.com/mdsaifjawed/DilliGuide.git
cd DilliGuide
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```env
# Primary Google Gemini API Key (Required)
API_Key="your_primary_gemini_api_key_here"

# Backup Keys for Zero-Downtime Failover (Optional but Recommended)
API_Key_B1="your_first_backup_gemini_api_key_here"
API_Key_B2="your_second_backup_gemini_api_key_here"

# TinyFish Live Web Engine & Autonomous Agent Key (Optional)
Tinyfish_API="your_tinyfish_api_key_here"
```

---

## 🎯 Running the Project

### 1. Starting the FastAPI Server & Web Companion
Launch the backend server with hot-reload enabled:
```bash
python api.py
```
Or with Uvicorn directly:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Once started:
- **Interactive Companion UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Diagnostics**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 2. Interactive Terminal CLI Mode
DilliDarshan AI includes a rich terminal interface with ASCII visuals, live spinners, diagnostic panels, and multi-turn chat memory:
```bash
python retrieval.py
```
#### CLI Commands inside the Terminal Session:
- `history` or `session`: Displays all turns in the active multi-turn conversation.
- `reset` or `new`: Clears active conversational history.
- `stats`: Shows ChromaDB document counts, embedding dimensions, and active Gemini key.
- `exit` or `quit`: Exits the terminal CLI.

---

### 3. Ingestion & Database Management

#### A. Ingest a Single PDF or Markdown File:
```bash
python ingesting.py data/lalquila.pdf
```

#### B. Ingest an Entire Directory of Documents:
```bash
python ingesting.py data/
```

#### C. Autonomously Research and Ingest a New Delhi Landmark via TinyFish:
```bash
python ingesting.py --tinyfish "Lotus Temple"
```
*What happens:* TinyFish searches the web for Lotus Temple, extracts clean Markdown, structures it into a verified 5-section guide with Gemini, and ingests it into `ChromaDB` automatically.

#### D. Reset Database and Re-ingest Baseline Corpus:
```bash
python ingesting.py --reingest
```

#### E. Clear the Vector Collection:
```bash
python ingesting.py --clear
```

---

### 4. Automated Backend & Frontend Verification Suite
Run the automated test suite to verify all endpoints, multi-turn place inheritance, HTML formatting, and SQLite persistence:
```bash
python verify_backend.py
```
*Test suite validates:*
- `GET /health` diagnostics
- `GET /` companion page serving
- `POST /user/profile` upsert and case-insensitive SQLite index
- `POST /chat` Turn 1 (Red Fort timings & tickets)
- `POST /chat` Turn 2 (Pronoun context retention: "*Which metro gate is closest to reach there?*")
- Multi-lingual Hindi response generation
- `POST /chat/reset` session management
- SQLite `chat_logs` audit trail entries

---

## ⚙️ Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|:--------:|---------|
| `API_Key` | Primary Gemini API Key for Embeddings and Generation | **Yes** | `AIzaSy...` |
| `API_Key_B1` | First Failover Key (activated on 429 quota exhaustion) | No | `AIzaSy...` |
| `API_Key_B2` | Second Failover Key (activated if B1 fails) | No | `AIzaSy...` |
| `Tinyfish_API` | TinyFish API key for live web search and browser agent | No | `tf_live_...` |

---

## 🛠 Tech Stack

- **Core Language**: Python 3.10+
- **Web API Framework**: FastAPI, Uvicorn, Pydantic v2
- **Vector Database**: ChromaDB (Cosine similarity / HNSW index)
- **Large Language Models**: Google Gemini 3.8 Flash, 3.6 Flash, 2.5 Flash (`google-genai` SDK)
- **Embedding Models**: Google `gemini-embedding-2`
- **Live Web Research**: TinyFish Python SDK (`tinyfish`)
- **Document Processing**: `pymupdf4llm`, `langchain-text-splitters`
- **Relational Storage**: SQLite3 (with WAL mode & foreign keys)
- **Terminal UI**: Rich (Color formatting, Tables, Panels, Markdown)
- **Companion Frontend**: Semantic HTML5, Vanilla CSS3, Modern ES6 JavaScript

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

```text
MIT License
Copyright (c) 2026 MD SAIF JAWED
```

---

## 👨‍💻 Author & Acknowledgments

- **Developed by**: [MD SAIF JAWED](https://github.com/mdsaifjawed)
- **Repository**: [mdsaifjawed/Dilli-Darshan](https://github.com/mdsaifjawed/Dilli-Darshan.git)
- **Special Thanks**:
  - [Google Gemini Team](https://ai.google.dev/) for high-throughput multimodal & embedding models.
  - [ChromaDB](https://www.trychroma.com/) for fast, local-first vector embeddings storage.
  - [TinyFish](https://tinyfish.io/) for high-speed live web search and agentic browsing capabilities.
  - Delhi Tourism & Archaeological Survey of India (ASI) for heritage reference records.
