import os
import chromadb
from dotenv import load_dotenv
from google import genai as gr
from google.genai import types
from tinyfish import TinyFish

load_dotenv()

# 1. Initialize Gemini Client
api_key = os.getenv("API_Key")
if not api_key:
    raise ValueError("API_Key not found in environment variables. Please check your .env file.")

client = gr.Client(api_key=api_key)

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

