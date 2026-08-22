import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index("faiss_index.index")

# -----------------------------
# Load knowledge chunks
# -----------------------------
with open("knowledge_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, top_k=5):

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


# -----------------------------
# Test retrieval
# -----------------------------
query = """
I need to learn Python, SQL, Data Science, Machine Learning,
Data Mining, Predictive Modeling, Data Visualization,
Big Data, Apache Spark, Hadoop, AWS and Tableau
for a Data Scientist career.
"""
results = retrieve(query, top_k=5)

print("=" * 60)
print("QUERY:")
print(query)

print("\nRETRIEVED KNOWLEDGE:")
print("=" * 60)

for i, result in enumerate(results, 1):

    print(f"\nResult {i}")
    print("Score:", result["score"])
    print("Content:")
    print(result["content"][:1000])
    print("-" * 60)