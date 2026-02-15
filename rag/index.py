# rag/index.py

import faiss
import numpy as np


def build_faiss_index(embeddings):
    embeddings = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding, top_k):
    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    return scores[0], indices[0]
