from src.datasets.caltech101 import get_dataset

dataset, dataloader = get_dataset()

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)