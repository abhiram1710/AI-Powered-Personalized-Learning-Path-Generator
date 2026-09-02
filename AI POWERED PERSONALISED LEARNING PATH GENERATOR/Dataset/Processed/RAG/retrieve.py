import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
INDEX_PATH = RAG_DIR / "faiss_index.index"
CHUNKS_PATH = RAG_DIR / "knowledge_chunks.pkl"


def load_assets():
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as handle:
        chunks = pickle.load(handle)
    return index, chunks, SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, top_k=5):
    index, chunks, embedding_model = load_assets()

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(distance),
            "content": chunks[idx]["content"]
        })

    return results
