import torch
from torchvision.models import ResNet152_Weights, resnet152


class Overfit(torch.nn.Module):
    prime: torch.nn.Parameter  # type: ignore
    num_classes: int

    def __init__(
        self,
        pretrained_classifier: torch.nn.Module = resnet152(
            weights=ResNet152_Weights.IMAGENET1K_V1
        ),
        num_classes=1000,
    ):
        super().__init__()
        self.pretrained_classifier = pretrained_classifier.eval()
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.prime = torch.nn.Parameter(  # type: ignore
            torch.zeros(self.num_classes), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_src = self.pretrained_classifier(x)
        # y_tgt = torch.nn.functional.softmax(y_src, dim=1) + self.prime
        y_tgt = y_src + self.prime
        return y_tgt
