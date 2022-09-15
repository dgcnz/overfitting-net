import math
from typing import List, Optional

import mlflow
import torch
import torch.nn.functional as F
from mlflow import log_metric
from torchvision.models import ResNet152_Weights, resnet152

from overfit.models.overfit import Overfit
from overfit.utils.mlflow import log_idx, log_max, log_norm


class OverfitTrainer:
    def __init__(self):
        pass

    def set(
        self,
        pretrained_classifier=resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
        num_classes=1000,
        confidence=0.1,
        weight_decay=30,
        max_lr=0.1,
        momentum=0.9,
        initial_lr=0.05,
    ):
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.model = Overfit(pretrained_classifier, num_classes, confidence)
        self.max_lr = max_lr
        from torch.optim import SGD

        # from overfit.optimizers.sgdmod import SGDMod

        self.optimizer = SGD(
            [self.model.prime],
            lr=initial_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    def forward_backward(
        self, x: torch.Tensor, y_ix: Optional[int] = None, step: Optional[int] = None
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_src = self.model.pretrained_classifier(x)
        y_tgt = self.model(x)
        y_pseudo = self.model.pseudo_ground_truth(y_tgt)
        prime = self.model.prime

        # Normalized to )0, 1)
        H_src = torch.distributions.Categorical(logits=y_src).entropy()[0] / math.log(
            self.model.num_classes
        )
        new_lr = self.max_lr * (1 - H_src)
        assert new_lr >= 0.0
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)
        p_y_tgt = F.softmax(y_tgt, dim=1)
        p_y_src = F.softmax(y_src, dim=1)
        # y_tgt_ix = torch.argmax(p_y_tgt, dim=1)

        # START logging

        if step is not None:
            if y_ix is not None:
                log_idx("Correct probability", p_y_tgt[0], y_ix, step)
                log_idx("Correct prime", prime, y_ix, step)
                log_idx("Source Correct Prediction", p_y_src[0], y_ix, step)
                log_idx("Target Correct Prediction", p_y_tgt[0], y_ix, step)
                log_idx("Source Unnormalized Correct Prediction", y_src[0], y_ix, step)
                log_idx("Target Unnormalized Correct Prediction", y_tgt[0], y_ix, step)

            log_norm("Target Prediction Norm", p_y_tgt, step)
            log_norm("Target Unnormalized Prediction Norm", y_tgt, step)
            log_max("Target Prediction Max", p_y_tgt, step)
            log_max("Target Unnormalized Prediction Max", y_tgt, step)
            log_max("Source Unnormalized Prediction Max", y_src, step)
            log_metric("Source Prediction Entropy", H_src, step)
            log_metric("Learning Rate", new_lr, step)
            log_norm("Prime Norm", prime, step)
            log_max("Prime Max", prime, step)

        # END logging

        # UPDATE
        self.update(new_lr, pseudo_loss)
        # self.model.prime = torch.nn.Parameter(  # type: ignore
        #     torch.clamp(self.model.prime, 0, 1)
        # )
        return y_tgt

    def update(self, new_lr, pseudo_loss):
        for ix, _ in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ix]["lr"] = new_lr
        pseudo_loss.backward()
        self.optimizer.step()

    def new_experiment(self, experiment_name="test"):
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.pytorch.log_model(self.model, "overfit")
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("lr_scale", self.max_lr)
        mlflow.log_param("momentum", self.momentum)

    def test(self, X: List[torch.Tensor], Y: List[int]):
        n = len(X)
        assert n == len(Y)
        for step, x, y in zip(range(n), X, Y):
            self.forward_backward(x, y, step)
