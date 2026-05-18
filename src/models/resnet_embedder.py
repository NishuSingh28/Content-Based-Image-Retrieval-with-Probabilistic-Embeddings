import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ResNetEmbedder:
    def __init__(self, device=None):
        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Load pretrained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove classification layer
        self.model = nn.Sequential(*list(model.children())[:-1])

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    @torch.no_grad()
    def extract(self, image_path):
        image_tensor = self.preprocess(image_path)

        embedding = self.model(image_tensor)

        # Flatten embedding
        embedding = embedding.view(embedding.size(0), -1)

        # Normalize embedding
        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.cpu().numpy()
    
    @torch.no_grad()

    def extract_tensor(self, image_tensor):

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        embedding = self.model(image_tensor)

        embedding = embedding.view(
            embedding.size(0),
            -1
        )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.cpu().numpy()