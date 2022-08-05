import torch

from mlflow import log_metric


def log_max(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.max(x).detach().numpy(), step)


def log_norm(name: str, x: torch.Tensor, step: int):
    log_metric(name, torch.norm(x), step)


def log_idx(name: str, x: torch.Tensor, idx: int, step: int):
    log_metric(name, x.detach().numpy()[idx], step)
