import torch

from mlflow import log_metric
from mlflow.entities import Metric


def log_max(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.max(x).cpu().detach().numpy(), step)


def log_norm(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.norm(x).cpu(), step)


def log_idx(name: str, x: torch.Tensor, idx: int, step: int, timestamp):
    log_metric(name, x.cpu().detach().numpy()[idx], step)


def get_log_max(name: str, x: torch.Tensor, step: int, timestamp: int):
    return Metric(
        key=name,
        value=float(torch.max(x).cpu().detach().numpy()),
        timestamp=timestamp,
        step=step,
    )


def get_log_norm(name: str, x: torch.Tensor, step: int, timestamp: int):
    return Metric(
        key=name, value=float(torch.norm(x).cpu()), timestamp=timestamp, step=step
    )


def get_log_idx(name: str, x: torch.Tensor, idx: int, step: int, timestamp: int):
    return Metric(
        key=name,
        value=float(x.detach().cpu().numpy()[idx]),
        timestamp=timestamp,
        step=step,
    )


def get_or_create_experiment_by_name(client, experiment_name: str) -> str:
    """
    Creates or fetches experiment by MLFLOW_EXPERIMENT_NAME
    Returns MLFLOW_EXPERIMENT_ID
    """
    try:
        return client.create_experiment(experiment_name)
    except Exception:
        return client.get_experiment_by_name(experiment_name).experiment_id


def get_experiment_name(
    dataset: str,
    model: str,
    confidence: float,
    weight_decay: float,
    max_lr: float,
    momentum: float,
):
    return f"D{dataset}M{model}C{confidence}WD{weight_decay}LR{max_lr}M{momentum}"
