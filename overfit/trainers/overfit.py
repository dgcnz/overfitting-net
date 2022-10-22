import math
from typing import List, Optional

import mlflow
import torch
import torch.nn.functional as F
from mlflow import log_metric
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

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
    ):
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.confidence = confidence
        self.model = Overfit(pretrained_classifier, num_classes)
        self.max_lr = max_lr
        from torch.optim import SGD

        # from overfit.optimizers.sgdmod import SGDMod

        self.optimizer = SGD(
            [self.model.prime],
            lr=0.0,  # doesn't matter as lr will be determined on runtime
            weight_decay=weight_decay,
            momentum=momentum,
        )

    # def pseudo_ground_truth(self, y_hat: torch.Tensor) -> torch.Tensor:
    #     k = math.e + self.confidence
    #     return math.log(k) * y_hat
    def sharpen(self, p, T, dim: int):
        p_T = torch.pow(p, 1 / T)
        return p_T / torch.sum(p_T, dim=dim, keepdim=True)

    def forward_backward(
        self, x: torch.Tensor, y_ix: Optional[int] = None, step: Optional[int] = None
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_src = self.model.pretrained_classifier(x)
        y_tgt = self.model(x)
        p_y_tgt = F.softmax(y_tgt, dim=1)
        y_pseudo = self.sharpen(p_y_tgt, 1 - self.confidence, dim=1)
        prime = self.model.prime

        # Normalized to )0, 1)
        H_src = torch.distributions.Categorical(logits=y_src).entropy()[0] / math.log(
            self.model.num_classes
        )
        new_lr = self.max_lr * (1 - H_src)
        assert new_lr >= 0.0
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)
        p_y_src = F.softmax(y_src, dim=1)
        p_y_src_ix = torch.argmax(p_y_src, dim=1)
        p_y_tgt_ix = torch.argmax(p_y_tgt, dim=1)

        # START logging

        if step is not None:
            if y_ix is not None:
                log_idx("Correct probability", p_y_tgt[0], y_ix, step)
                log_idx("Correct prime", prime, y_ix, step)
                log_idx("Source Normalized Correct Prediction", p_y_src[0], y_ix, step)
                log_idx("Target Normalized Correct Prediction", p_y_tgt[0], y_ix, step)
                log_idx("Source Unnormalized Correct Prediction", y_src[0], y_ix, step)
                log_idx("Target Unnormalized Correct Prediction", y_tgt[0], y_ix, step)
                log_idx(
                    "Source Normalized Prediction", p_y_src[0], int(p_y_src_ix), step
                )
                log_idx(
                    "Target Normalized Prediction", p_y_tgt[0], int(p_y_tgt_ix), step
                )

                def rank(x: torch.Tensor, ix: int):
                    return (torch.argsort(x, descending=True) == ix).nonzero().item()

                mlflow.log_metric("Target Correct Rank", rank(p_y_tgt[0], y_ix), step)
                mlflow.log_metric("Source Correct Rank", rank(p_y_src[0], y_ix), step)

            log_norm("Target Normalized Prediction Norm", p_y_tgt, step)
            log_norm("Target Unnormalized Prediction Norm", y_tgt, step)
            log_max("Target Normalized Prediction Max", p_y_tgt, step)
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

    def test(self, X: List[torch.Tensor], Y: List[int], categories: List[str]):
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("max_lr", self.max_lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("confidence", self.confidence)
        n = len(X)
        assert n == len(Y)

        tgt_preds_txt = ""
        src_preds_txt = ""
        for step, x, y in tqdm(list(zip(range(n), X, Y))):
            tgt_pred = self.forward_backward(x, y, step)
            src_pred = self.model.pretrained_classifier.forward(x)
            tgt_cat = categories[tgt_pred[0].argmax().item()]  # type: ignore
            src_cat = categories[src_pred[0].argmax().item()]  # type: ignore
            tgt_preds_txt += f"[{step + 1}] {tgt_cat}\n"
            src_preds_txt += f"[{step + 1}] {src_cat}\n"
        mlflow.log_text(tgt_preds_txt, "target_predictions.txt")
        mlflow.log_text(src_preds_txt, "source_predictions.txt")
