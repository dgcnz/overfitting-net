import torch

from mlflow import log_metric
from mlflow.entities import Metric


def log_max(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.max(x).detach().numpy(), step)


def log_norm(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.norm(x), step)


def log_idx(name: str, x: torch.Tensor, idx: int, step: int, timestamp):
    log_metric(name, x.detach().numpy()[idx], step)


def get_log_max(name: str, x: torch.Tensor, step: int, timestamp: int):
    return Metric(
        key=name,
        value=float(torch.max(x).detach().numpy()),
        timestamp=timestamp,
        step=step,
    )


def get_log_norm(name: str, x: torch.Tensor, step: int, timestamp: int):
    return Metric(key=name, value=float(torch.norm(x)), timestamp=timestamp, step=step)


def get_log_idx(name: str, x: torch.Tensor, idx: int, step: int, timestamp: int):
    return Metric(
        key=name, value=float(x.detach().numpy()[idx]), timestamp=timestamp, step=step
    )
