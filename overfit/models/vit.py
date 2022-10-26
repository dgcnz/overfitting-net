import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification


class ViT(torch.nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
        )

    def forward(self, X):
        Z = self.feature_extractor(X, return_tensors="pt")
        return self.model(**Z).logits  # type: ignore
