import streamlit as st
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.datasets.caltech101 import (
    get_dataset,
    get_label_name
)

from src.models.resnet_embedder import ResNetEmbedder
from src.retrieval.retriever import ImageRetriever


st.set_page_config(
    page_title="CBIR System",
    layout="wide"
)

st.title("Content-Based Image Retrieval")

st.write(
    "Retrieve visually similar images using "
    "ResNet50 embeddings."
)

# Load dataset
@st.cache_resource
def load_system():

    dataset, _ = get_dataset()

    embedder = ResNetEmbedder()

    retriever = ImageRetriever()

    for i in range(200):

        image_tensor, label = dataset[i]

        embedding = embedder.extract_tensor(
            image_tensor
        )

        retriever.add(
            embedding,
            label,
            image_tensor
        )

    retriever.build()

    return dataset, embedder, retriever


dataset, embedder, retriever = load_system()

# Query image selection
query_index = st.slider(
    "Select Query Image",
    0,
    len(dataset) - 1,
    0
)

query_tensor, query_label = dataset[query_index]

query_embedding = embedder.extract_tensor(
    query_tensor
)

results = retriever.search(
    query_embedding,
    top_k=5
)

st.subheader(
    f"Query: "
    f"{get_label_name(dataset, query_label)}"
)

# Display query image
st.image(
    TF.to_pil_image(query_tensor),
    width=250
)

st.subheader("Retrieved Images")

cols = st.columns(5)

for col, result in zip(cols, results):

    with col:

        label_name = get_label_name(
            dataset,
            result["label"]
        )

        st.image(
            TF.to_pil_image(result["image"]),
            use_container_width=True
        )

        st.write(label_name)

        st.write(
            f"Similarity: "
            f"{result['similarity']:.2f}"
        )