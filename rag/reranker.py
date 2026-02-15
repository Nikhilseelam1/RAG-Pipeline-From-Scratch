from sentence_transformers import CrossEncoder


def load_reranker(model_name):
    reranker = CrossEncoder(model_name)
    return reranker


def rerank_results(query, context_items, reranker, top_k):
    pairs = [[query, item["sentence_chunk"]] for item in context_items]

    rerank_scores = reranker.predict(pairs)

    scored_items = list(zip(rerank_scores, context_items))
    scored_items.sort(key=lambda x: x[0], reverse=True)

    top_items = scored_items[:top_k]

    reranked_contexts = [item[1] for item in top_items]

    return reranked_contexts
