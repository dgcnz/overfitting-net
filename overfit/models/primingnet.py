import math
from typing import Optional

import torch
import torch.nn.functional as F
from mlflow import log_metric
from torchvision.models import ResNet152_Weights, resnet152


class PrimingNet(torch.nn.Module):
    def __init__(
        self,
        pretrained_classifier=resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
        num_classes=1000,
        confidence=0.1,
        weight_decay=0.1,
        lr_scale=1,
        momentum=0.9,
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
        self.optimizer = torch.optim.SGD(
            [self.prime], lr=0, weight_decay=weight_decay, momentum=momentum
        )
        self.confidence = confidence

    def pseudo_ground_truth(self, y_hat: torch.Tensor) -> torch.Tensor:
        k = math.e + self.confidence
        return math.log(k) * y_hat

    def forward(
        self, x: torch.Tensor, y: Optional[int] = None, step: Optional[int] = None
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_src = self.pretrained_classifier(x)
        y_tgt = y_src + self.prime
        y_pseudo = self.pseudo_ground_truth(y_tgt)
        H_src = torch.distributions.Categorical(logits=y_src).entropy()[0] / math.log2(
            self.num_classes
        )
        new_lr = self.lr_scale * (1 - H_src)

        # START logging

        prob = F.softmax(y_tgt, dim=1)

        # top5_prob, top5_catid = torch.topk(probabilities, 5)
        if step is not None:
            if y is not None:
                log_metric("Correct probability", prob.detach().numpy()[0][y], step)
                log_metric("Correct prime", self.prime.detach().numpy()[y], step)
            log_metric("Target Prediction Norm", torch.norm(prob), step)
            log_metric("Target Prediction Max", torch.max(prob).detach().numpy(), step)
            log_metric("Target Unnormalized Prediction Norm", torch.norm(y_tgt), step)
            log_metric(
                "Target Unnormalized Prediction Max",
                torch.max(y_tgt).detach().numpy(),
                step,
            )
            log_metric(
                "Source Unnormalized Prediction Max",
                torch.max(y_src).detach().numpy(),
                step,
            )
            log_metric(
                "Source Correct Prediction",
                F.softmax(y_src, dim=1).detach().numpy()[0][y],
                step,
            )

            log_metric(
                "Target Correct Prediction",
                F.softmax(y_tgt, dim=1).detach().numpy()[0][y],
                step,
            )
            log_metric("Source Prediction Entropy", H_src, step)
            log_metric("Learning Rate", new_lr, step)
            log_metric("Prime Norm", torch.norm(self.prime), step)
            log_metric("Prime Max", torch.max(self.prime).detach().numpy(), step)

        # END logging
        for ix, _ in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ix]["lr"] = new_lr
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)
        pseudo_loss.backward()
        self.optimizer.step()
        return y_tgt
