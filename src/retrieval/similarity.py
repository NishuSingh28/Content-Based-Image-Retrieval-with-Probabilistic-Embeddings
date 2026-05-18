import numpy as np


def cosine_similarity(query_embedding, embeddings):

    query_norm = query_embedding / np.linalg.norm(query_embedding)

    embeddings_norm = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    similarities = np.dot(embeddings_norm, query_norm.T)

    return similarities.squeeze()