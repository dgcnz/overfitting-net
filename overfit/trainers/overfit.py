import logging
from typing import List, Optional, Union

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
    step: int = 0
    src_acc_top1: int = 0
    tgt_acc_top1: int = 0
    src_acc_top5: int = 0
    tgt_acc_top5: int = 0
    tgt_preds_txt: str = ""
    src_preds_txt: str = ""

    def __init__(self, categories: List[str]):
        self.categories = categories

    def set(
        self,
        pretrained_classifier: torch.nn.Module = resnet152(
            weights=ResNet152_Weights.IMAGENET1K_V1
        ),
        num_classes=1000,
        confidence=0.1,
        weight_decay=0.1,
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
        self.src_acc_top1 = 0
        self.tgt_acc_top1 = 0
        self.src_acc_top5 = 0
        self.tgt_acc_top5 = 0
        self.step = 0

    def send_logs(self, active_run: mlflow.ActiveRun):
        logging.info(mlflow.get_artifact_uri())
        logging.info(mlflow.get_tracking_uri())
        logging.info(mlflow.get_registry_uri())
        client = MlflowClient()
        client.log_text(
            run_id=active_run.info.run_id,
            text=self.src_preds_txt,
            artifact_file="source_predictions.txt",
        )
        client.log_text(
            run_id=active_run.info.run_id,
            text=self.tgt_preds_txt,
            artifact_file="target_predictions.txt",
        )
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
        step = self.step
        p_y_src_ix = int(torch.argmax(p_y_src, dim=1).item())
        p_y_tgt_ix = int(torch.argmax(p_y_tgt, dim=1).item())
        p_src_rank = rank(p_y_src[0], y_ix)
        p_tgt_rank = rank(p_y_tgt[0], y_ix)
        assert isinstance(y_ix, int)
        assert isinstance(p_y_src_ix, int)
        assert isinstance(p_y_tgt_ix, int)
        timestamp = step
        self.tgt_acc_top1 += p_y_tgt_ix == y_ix
        self.src_acc_top1 += p_y_src_ix == y_ix
        self.tgt_acc_top5 += p_src_rank < 5
        self.src_acc_top5 += p_tgt_rank < 5
        self.metric_history += [
            Metric(
                key="Source Accumulated Top-1 Count",
                value=self.src_acc_top1,
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                key="Target Accumulated Top-1 Count",
                value=self.tgt_acc_top1,
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                key="Source Accumulated Top-5 Count",
                value=self.src_acc_top1,
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                key="Target Accumulated Top-5 Count",
                value=self.tgt_acc_top1,
                step=step,
                timestamp=timestamp,
            ),
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
                p_tgt_rank,
                step=step,
                timestamp=timestamp,
            ),
            Metric(
                "Source Correct Rank",
                p_src_rank,
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

        tgt_cat = self.categories[p_y_tgt_ix]
        src_cat = self.categories[p_y_src_ix]
        self.tgt_preds_txt += f"[{step + 1}] {tgt_cat}\n"
        self.src_preds_txt += f"[{step + 1}] {src_cat}\n"
        self.step += 1

    def forward_backward(
        self, x: Union[torch.Tensor, List[torch.Tensor]], y_ix: Optional[int] = None
    ) -> torch.Tensor:
        assert len(x) == 1  # don't handle batched processing
        self.optimizer.zero_grad()
        y_src = self.model.pretrained_classifier(x)
        y_tgt = self.model(x)
        p_y_src = F.softmax(y_src, dim=1)
        p_y_tgt = F.softmax(y_tgt, dim=1)
        y_pseudo = sharpen(p_y_tgt, 1 - self.confidence, dim=1)
        # y_pseudo = F.softmax(p_y_tgt, dim=1)
        pseudo_loss = F.cross_entropy(y_tgt, y_pseudo)

        H_src = entropy(logits=y_src)[0].item()
        new_lr = self.max_lr * (1 - H_src)
        assert 0.0 <= H_src and H_src <= 1.0
        assert new_lr >= 0.0

        if y_ix is not None:
            self.log_metrics(
                y_ix=y_ix,
                p_y_src=p_y_src.cpu(),
                p_y_tgt=p_y_tgt.cpu(),
                y_src=y_src.cpu(),
                y_tgt=y_tgt.cpu(),
                prime=self.model.prime.cpu(),
                new_lr=float(new_lr),
                H_src=float(H_src),
            )

        self.update(new_lr, pseudo_loss)
        return y_tgt

    def update(self, new_lr, pseudo_loss):
        for ix, _ in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ix]["lr"] = new_lr
        pseudo_loss.backward()
        self.optimizer.step()

    def test(
        self,
        X: torch.Tensor,
        Y: List[int],
        active_run: mlflow.ActiveRun,
        hf_format: bool = False,
    ):
        """
        X: video tensor on (T, C, H, W) format
        """
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("max_lr", self.max_lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("confidence", self.confidence)
        assert len(X) == len(Y)

        for x, y in list(zip(X, Y)):
            if hf_format:
                _ = self.forward_backward([x], y)
            else:
                _ = self.forward_backward(x.unsqueeze(0), y)
        self.send_logs(active_run=active_run)
