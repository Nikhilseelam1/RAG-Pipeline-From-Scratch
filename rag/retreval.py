from time import perf_counter as timer


def retrieve_relevant_resources(
    query,
    embedding_model,
    index,
    pages_and_chunks,
    top_k
):
    query_embedding = embedding_model.encode(query)

    start_time = timer()
    scores, indices = index.search(
        query_embedding.reshape(1, -1).astype("float32"),
        top_k
    )
    end_time = timer()

    print(f"[INFO] FAISS search time over {index.ntotal} vectors: {end_time - start_time:.5f}s")

    context_items = [pages_and_chunks[i] for i in indices[0]]

    return scores[0], indices[0], context_items
