import os
import chromadb
from dotenv import load_dotenv
from google import genai as gr

load_dotenv()

# 1. Initialize Gemini Client
api_key = os.getenv("API_Key")
if not api_key:
    raise ValueError("API_Key not found in environment variables. Please check your .env file.")

client = gr.Client(api_key=api_key)

# 2. Initialize ChromaDB Persistent Client
# The database will be stored on disk in the 'chroma_db' folder
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 3. Create or load collection
collection = chroma_client.get_or_create_collection(
    name="dilli_guide_collection",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str):
    """Generate embedding vector for a single text using Gemini model."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return response.embeddings[0].values

def get_embeddings(texts: list[str]):
    """Generate embedding vectors for multiple texts in a single batch API call."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=texts
    )
    return [e.values for e in response.embeddings]

def add_document(doc_id: str, text: str, metadata: dict = None):
    """Generate embedding and persist a single document into ChromaDB."""
    embedding = get_embedding(text)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata] if metadata else None
    )

def add_documents(doc_ids: list[str], texts: list[str], metadatas: list[dict] = None):
    """Batch generate embeddings and persist multiple documents into ChromaDB."""
    embeddings = get_embeddings(texts)
    collection.upsert(
        ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas if metadatas else None
    )

def clear_database():
    """Wipes all documents and resets the persistent ChromaDB collection."""
    global collection
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

