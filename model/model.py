import math

import torch
import torch.nn.functional as F
import torchvision
from base import BaseModel


class PrimingNet(torch.nn.Module):
    def __init__(
        self,
        pretrained_classifier=torchvision.models.convnext_tiny(pretrained=True),
        num_classes=1000,
        confidence=1,
    ):
        super().__init__()
        self.pretrained_classifier = pretrained_classifier.eval()
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.prime = torch.nn.Parameter(  # type: ignore
            torch.zeros(self.num_classes), requires_grad=True
        )
        self.initial_lr = 0.05
        self.optimizer = torch.optim.SGD(
            [self.prime], lr=self.initial_lr, weight_decay=0.1
        )
        self.confidence = confidence

    def pseudo_ground_truth(self, y_hat: torch.Tensor) -> torch.Tensor:
        k = math.e + self.confidence
        return math.log(k) * y_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_hat = self.pretrained_classifier(x) + self.prime
        y_pseudo = self.pseudo_ground_truth(y_hat)
        pseudo_loss = F.cross_entropy(y_hat, y_pseudo)
        pseudo_loss.backward()
        self.optimizer.step()
        return y_hat
