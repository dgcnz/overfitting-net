import logging
import time
from typing import List, Optional

import mlflow
import torch
import torch.nn.functional as F
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from torch.optim import SGD
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from overfit.models.overfit import Overfit
from overfit.utils.misc import batch, entropy, rank, sharpen
from overfit.utils.mlflow import get_log_idx, get_log_max, get_log_norm


class OverfitTrainer:
    categories: List[str]
    metric_history: List[Metric] = []
    tgt_preds_txt: str = ""
    src_preds_txt: str = ""

    def __init__(self, categories: List[str]):
        self.categories = categories

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
        self.optimizer = SGD(
            [self.model.prime],
            lr=0.0,  # doesn't matter as lr will be determined on runtime
            weight_decay=weight_decay,
            momentum=momentum,
        )

    def reset_logs_history(self):
        self.metric_history.clear()
        self.src_preds_txt = ""
        self.tgt_preds_txt = ""

    def send_logs(self, active_run: mlflow.ActiveRun):
        client = MlflowClient()
        mlflow.log_text(self.tgt_preds_txt, "target_predictions.txt")
        mlflow.log_text(self.src_preds_txt, "source_predictions.txt")
        logging.info("Uploading logs")
        for logs in batch(self.metric_history, 1000):
            client.log_batch(run_id=active_run.info.run_id, metrics=logs)
        self.reset_logs_history()

    def log_metrics(
        self,
        y_ix: int,
        p_y_src: torch.Tensor,
        p_y_tgt: torch.Tensor,
        y_src: torch.Tensor,
        y_tgt: torch.Tensor,
        prime: torch.Tensor,
        H_src: float,
        new_lr: float,
    ) -> None:
        """Log metrics to mlflow."""
        step = len(self.metric_history)
        p_y_src_ix = torch.argmax(p_y_src, dim=1)
        p_y_tgt_ix = torch.argmax(p_y_tgt, dim=1)
        timestamp = int(time.time())
        self.metric_history += [
            get_log_idx(
                "Correct probability", p_y_tgt[0], y_ix, step=step, timestamp=timestamp
            ),
            get_log_idx("Correct prime", prime, y_ix, step=step, timestamp=timestamp),
            get_log_idx(
                "Source Normalized Correct Prediction",
                p_y_src[0],
                y_ix,
                step=step,
                timestamp=timestamp,
            ),
            get_log_idx(
                "Target Normalized Correct Prediction",
                p_y_tgt[0],
                y_ix,
                step=step,
                timestamp=timestamp,
            ),
            get_log_idx(
                "Source Unnormalized Correct Prediction",
                y_src[0],
                y_ix,
                step=step,
                timestamp=timestamp,
            ),
            get_log_idx(
                "Target Unnormalized Correct Prediction",
                y_tgt[0],
                y_ix,
                step=step,
                timestamp=timestamp,
            ),
            get_log_idx(
                "Source Normalized Prediction",
                p_y_src[0],
                int(p_y_src_ix),
                step=step,
                timestamp=timestamp,
            ),
            get_log_idx(
                "Target Normalized Prediction",
                p_y_tgt[0],
                int(p_y_tgt_ix),
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                "Target Correct Rank",
                rank(p_y_tgt[0], y_ix),
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                "Source Correct Rank",
                rank(p_y_src[0], y_ix),
                step=step,
                timestamp=timestamp,
            ),
            get_log_norm(
                "Target Normalized Prediction Norm",
                p_y_tgt,
                step=step,
                timestamp=timestamp,
            ),
            get_log_norm(
                "Target Unnormalized Prediction Norm",
                y_tgt,
                step=step,
                timestamp=timestamp,
            ),
            get_log_max(
                "Target Normalized Prediction Max",
                p_y_tgt,
                step=step,
                timestamp=timestamp,
            ),
            get_log_max(
                "Target Unnormalized Prediction Max",
                y_tgt,
                step=step,
                timestamp=timestamp,
            ),
            get_log_max(
                "Source Unnormalized Prediction Max",
                y_src,
                step=step,
                timestamp=timestamp,
            ),
            Metric("Source Prediction Entropy", H_src, step=step, timestamp=timestamp),
            Metric("Learning Rate", new_lr, step=step, timestamp=timestamp),
            get_log_norm("Prime Norm", prime, step=step, timestamp=timestamp),
            get_log_max("Prime Max", prime, step=step, timestamp=timestamp),
        ]

        tgt_cat = self.categories[int(p_y_tgt_ix.item())]
        src_cat = self.categories[int(p_y_src_ix.item())]
        self.tgt_preds_txt += f"[{step + 1}] {tgt_cat}\n"
        self.src_preds_txt += f"[{step + 1}] {src_cat}\n"

    def forward_backward(
        self, x: torch.Tensor, y_ix: Optional[int] = None
    ) -> torch.Tensor:
        assert x.shape[0] == 1  # don't handle batched processing
        self.optimizer.zero_grad()
        y_src = self.model.pretrained_classifier(x)
        y_tgt = self.model(x)
        p_y_src = F.softmax(y_src, dim=1)
        p_y_tgt = F.softmax(y_tgt, dim=1)
        y_pseudo = sharpen(p_y_tgt, 1 - self.confidence, dim=1)
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)

        H_src = entropy(logits=y_src)[0].item()
        new_lr = self.max_lr * (1 - H_src)
        assert 0.0 <= H_src and H_src <= 1.0
        assert new_lr >= 0.0

        if y_ix is not None:
            self.log_metrics(
                y_ix=y_ix,
                p_y_src=p_y_src,
                p_y_tgt=p_y_tgt,
                y_src=y_src,
                y_tgt=y_tgt,
                prime=self.model.prime,
                new_lr=new_lr,
                H_src=H_src,
            )

        self.update(new_lr, pseudo_loss)
        return y_tgt

    def update(self, new_lr, pseudo_loss):
        for ix, _ in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ix]["lr"] = new_lr
        pseudo_loss.backward()
        self.optimizer.step()

    def test(self, X: torch.Tensor, Y: List[int], active_run: mlflow.ActiveRun):
        """
        X: video tensor on (T, C, H, W) format
        """
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("max_lr", self.max_lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("confidence", self.confidence)
        assert len(X) == len(Y)

        for x, y in tqdm(list(zip(X, Y))):
            _ = self.forward_backward(x.unsqueeze(0), y)
        self.send_logs(active_run=active_run)
