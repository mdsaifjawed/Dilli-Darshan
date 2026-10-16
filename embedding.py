import os
import chromadb
from dotenv import load_dotenv
from google import genai as gr
from google.genai import types
from tinyfish import TinyFish

load_dotenv()

# 1. Initialize Gemini Client with API_Key -> API_Key_B1 Failover
class ResilientModels:
    def __init__(self, client_manager: "ResilientGeminiClient"):
        self._manager = client_manager

    def embed_content(self, *args, **kwargs):
        return self._manager.execute_with_fallback("embed_content", *args, **kwargs)

    def generate_content(self, *args, **kwargs):
        return self._manager.execute_with_fallback("generate_content", *args, **kwargs)

    def __getattr__(self, name: str):
        def method(*args, **kwargs):
            return self._manager.execute_with_fallback(name, *args, **kwargs)
        return method


class ResilientGeminiClient:
    """
    Manages primary and backup Gemini API clients.
    Automatically catches errors (e.g. 429 quota exhaustion, rate limits, network errors)
    on the primary API_Key and seamlessly falls back to API_Key_B1.
    """
    def __init__(self, key_entries: list[tuple[str, str]]):
        self._entries = key_entries
        self._clients: list[tuple[str, gr.Client]] = []
        for label, k in key_entries:
            try:
                c = gr.Client(api_key=k)
                self._clients.append((label, c))
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini client for {label}: {e}")

        if not self._clients:
            raise ValueError("No valid Gemini API keys could be initialized.")

        self._active_idx = 0
        self.models = ResilientModels(self)

    @property
    def active_label(self) -> str:
        return self._clients[self._active_idx][0]

    @property
    def active_client(self) -> gr.Client:
        return self._clients[self._active_idx][1]

    @property
    def backup_label(self) -> str | None:
        if len(self._clients) > 1:
            next_idx = (self._active_idx + 1) % len(self._clients)
            return self._clients[next_idx][0]
        return None

    def execute_with_fallback(self, func_name: str, *args, **kwargs):
        num_clients = len(self._clients)
        last_error = None
        start_idx = self._active_idx

        for offset in range(num_clients):
            current_idx = (start_idx + offset) % num_clients
            label, client_obj = self._clients[current_idx]

            try:
                target_func = getattr(client_obj.models, func_name)
                result = target_func(*args, **kwargs)
                # If we succeeded on a backup client, keep it as the active client
                if current_idx != self._active_idx:
                    self._active_idx = current_idx
                return result
            except Exception as e:
                last_error = e
                # Check if there is another client to failover to
                if offset < num_clients - 1:
                    next_idx = (start_idx + offset + 1) % num_clients
                    next_label = self._clients[next_idx][0]
                    err_summary = str(e).strip()
                    if len(err_summary) > 120:
                        err_summary = err_summary[:120] + "..."

                    try:
                        from rich.console import Console
                        _c = Console()
                        _c.print(
                            f"\n[bold yellow]⚠️  [Gemini Failover][/bold yellow] Call failed on [cyan]{label}[/cyan]: [dim]{err_summary}[/dim]\n"
                            f"   [bold green]↳ Automatically switching to backup key '{next_label}' and retrying...[/bold green]"
                        )
                    except Exception:
                        print(
                            f"\n[Gemini Failover] Call failed on {label}: {err_summary}\n"
                            f"   -> Automatically switching to backup key '{next_label}' and retrying..."
                        )

                    self._active_idx = next_idx
                    continue
                else:
                    # All available keys exhausted
                    break

        raise last_error

    def __getattr__(self, name: str):
        return getattr(self.active_client, name)


key_entries: list[tuple[str, str]] = []

primary_key = (
    os.getenv("API_Key")
    or os.getenv("API_KEY")
    or os.getenv("api_key")
)
if primary_key:
    key_entries.append(("API_Key", primary_key.strip().strip('"').strip("'")))

backup_key = (
    os.getenv("API_Key_B1")
    or os.getenv("API_KEY_B1")
    or os.getenv("api_key_b1")
)
if backup_key:
    clean_backup = backup_key.strip().strip('"').strip("'")
    if not primary_key or clean_backup != primary_key.strip().strip('"').strip("'"):
        key_entries.append(("API_Key_B1", clean_backup))

if not key_entries:
    raise ValueError(
        "Neither 'API_Key' nor 'API_Key_B1' found in environment variables. Please check your .env file."
    )

client = ResilientGeminiClient(key_entries)
api_key = key_entries[0][1]

# 2. Initialize TinyFish Client
tinyfish_api_key = (
    os.getenv("Tinyfish_API")
    or os.getenv("TINYFISH_API_KEY")
    or os.getenv("TinyFish_API_Key")
    or os.getenv("tinyfish_api_key")
    or os.getenv("TINYFISH_API")
    or os.getenv("TINYFISH_KEY")
)
tf_client = TinyFish(api_key=tinyfish_api_key) if tinyfish_api_key else None
tinyfish_client = tf_client  # Backward-compatible alias



# 2. Initialize ChromaDB Persistent Client
# The database will be stored on disk in the 'chroma_db' folder
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 3. Create or load collection
collection = chroma_client.get_or_create_collection(
    name="dilli_guide_collection",
    metadata={"hnsw:space": "cosine"}
)


def sanitize_metadata(meta: dict | None) -> dict:
    """Ensures metadata contains no None values (converts None to empty string)."""
    if not meta:
        return {}
    return {k: ("" if v is None else v) for k, v in meta.items()}


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate embedding vector for a single text using Gemini model with task_type."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return response.embeddings[0].values


def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Generate embedding vectors for multiple texts in a single batch API call."""
    if not texts:
        return []

    contents_list = [
        types.Content(parts=[types.Part.from_text(text=t)])
        for t in texts
    ]
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents_list,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return [e.values for e in response.embeddings]


def get_query_embedding(query_text: str) -> list[float]:
    """Generate query embedding vector using task_type='RETRIEVAL_QUERY'."""
    return get_embedding(query_text, task_type="RETRIEVAL_QUERY")


def add_document(doc_id: str, text: str, metadata: dict = None):
    """Generate embedding and persist a single document into ChromaDB."""
    embedding = get_embedding(text, task_type="RETRIEVAL_DOCUMENT")
    sanitized_meta = sanitize_metadata(metadata)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[sanitized_meta] if sanitized_meta else None
    )


def add_documents(doc_ids: list[str], texts: list[str], metadatas: list[dict] = None):
    """Batch generate embeddings and persist multiple documents into ChromaDB."""
    if not doc_ids:
        return

    embeddings = get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
    sanitized_metas = (
        [sanitize_metadata(m) for m in metadatas]
        if metadatas
        else None
    )
    collection.upsert(
        ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=sanitized_metas
    )


def cleanup_collections():
    """Deletes test collections (like test_sanitize) and ensures clean state."""
    for col in chroma_client.list_collections():
        if col.name != "dilli_guide_collection":
            try:
                chroma_client.delete_collection(col.name)
                print(f"Deleted test/orphan collection: '{col.name}'")
            except Exception:
                pass


def clear_database():
    """Wipes all documents, resets the persistent ChromaDB collection, and removes orphan test collections."""
    global collection
    cleanup_collections()
    try:
        chroma_client.delete_collection("dilli_guide_collection")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(
        name="dilli_guide_collection",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{collection.name}' has been cleared (0 documents).")
    return True

