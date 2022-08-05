import math

import torch
from torchvision.models import ResNet152_Weights, resnet152


class Overfit(torch.nn.Module):
    def __init__(
        self,
        pretrained_classifier=resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
        num_classes=1000,
        confidence=0.1,
    ):
        super().__init__()
        self.pretrained_classifier = pretrained_classifier.eval()
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.prime = torch.nn.Parameter(  # type: ignore
            torch.zeros(self.num_classes), requires_grad=True
        )
        self.confidence = confidence

    def pseudo_ground_truth(self, y_hat: torch.Tensor) -> torch.Tensor:
        k = math.e + self.confidence
        return math.log(k) * y_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_src = self.pretrained_classifier(x)
        y_tgt = y_src + self.prime
        return y_tgt
