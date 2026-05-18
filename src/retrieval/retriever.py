import numpy as np
from src.retrieval.similarity import cosine_similarity


class ImageRetriever:

    def __init__(self):
        self.embeddings = []
        self.labels = []

    def add(self, embedding, label, image_tensor):

        self.embeddings.append(embedding.squeeze())

        self.labels.append({
            "label": label,
            "image": image_tensor
        })

    def build(self):

        self.embeddings = np.array(self.embeddings)

    def search(self, query_embedding, top_k=5):

        similarities = cosine_similarity(
            query_embedding.squeeze(),
            self.embeddings
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:

            metadata = self.labels[idx]

            results.append({
                "label": metadata["label"],
                "image": metadata["image"],
                "similarity": float(similarities[idx])
            })

        return results