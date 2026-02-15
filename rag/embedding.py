from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name, device):
    embedding_model = SentenceTransformer(
        model_name_or_path=model_name,
        device=device
    )
    return embedding_model


def generate_embeddings(embedding_model, texts):
    embeddings = embedding_model.encode(
        texts,
        convert_to_tensor=True
    )
    return embeddings
