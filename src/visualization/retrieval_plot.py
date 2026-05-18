import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.datasets.caltech101 import get_label_name


def show_results(query_image, results, dataset):

    num_results = len(results)

    plt.figure(figsize=(15, 4))

    # Query image
    plt.subplot(1, num_results + 1, 1)

    plt.imshow(TF.to_pil_image(query_image))

    plt.title("Query")

    plt.axis("off")

    # Retrieval results
    for i, result in enumerate(results):

        plt.subplot(1, num_results + 1, i + 2)

        plt.imshow(
            TF.to_pil_image(result["image"])
        )

        label_name = get_label_name(
            dataset,
            result["label"]
        )

        plt.title(
            f'{label_name}\n'
            f'Sim: {result["similarity"]:.2f}'
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()