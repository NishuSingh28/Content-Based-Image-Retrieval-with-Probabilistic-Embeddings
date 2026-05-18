from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataset(data_dir="data/raw", batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataset, dataloader


def get_label_name(dataset, label):

    return dataset.classes[label]