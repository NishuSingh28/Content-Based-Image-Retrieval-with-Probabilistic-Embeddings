from src.datasets.caltech101 import (
    get_dataset,
    get_label_name
)
from src.models.resnet_embedder import ResNetEmbedder
from src.retrieval.retriever import ImageRetriever
from src.visualization.retrieval_plot import show_results


dataset, _ = get_dataset()

embedder = ResNetEmbedder()

retriever = ImageRetriever()

print("Building embedding database...")

# Build embedding database
for i in range(100):

    image_tensor, label = dataset[i]

    embedding = embedder.extract_tensor(image_tensor)

    retriever.add(
        embedding,
        label,
        image_tensor
    )

retriever.build()

print("Database built.")

# Query image
query_tensor, query_label = dataset[0]

query_embedding = embedder.extract_tensor(query_tensor)

results = retriever.search(query_embedding, top_k=5)

print(
    f'\nQuery Label: '
    f'{get_label_name(dataset, query_label)}'
)

print("\nTop Retrieval Results:\n")

for idx, result in enumerate(results):

    print(f"Result {idx+1}")

    print(
        "Label:",
        get_label_name(dataset, result["label"])
    )

    print("Similarity:", round(result["similarity"], 4))

    print("-" * 30)

show_results(
    query_tensor,
    results,
    dataset
)