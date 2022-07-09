import math

import torch
import torch.nn.functional as F
from torchvision.models import ResNet152_Weights, resnet152


class PrimingNet(torch.nn.Module):
    def __init__(
        self,
        pretrained_classifier=resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
        num_classes=1000,
        confidence=0.1,
        weight_decay=0.1,
        lr_scale=1,
    ):
        super().__init__()
        self.pretrained_classifier = pretrained_classifier.eval()
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.prime = torch.nn.Parameter(  # type: ignore
            torch.zeros(self.num_classes), requires_grad=True
        )
        self.lr_scale = lr_scale
        self.optimizer = torch.optim.SGD([self.prime], lr=0, weight_decay=weight_decay)
        self.confidence = confidence

    def pseudo_ground_truth(self, y_hat: torch.Tensor) -> torch.Tensor:
        k = math.e + self.confidence
        return math.log(k) * y_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_src = self.pretrained_classifier(x)
        y_tgt = y_src + self.prime
        y_pseudo = self.pseudo_ground_truth(y_tgt)
        H_src = torch.distributions.Categorical(logits=y_src).entropy()
        for ix, _ in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ix]["lr"] = self.lr_scale * H_src
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)
        pseudo_loss.backward()
        self.optimizer.step()
        return y_tgt
