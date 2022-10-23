import math
from pathlib import Path
from typing import Tuple

import torch


def floor_even(x: int) -> int:
    """Rounds down integer to the nearest even integer."""
    return 2 * (x // 2)


def parse_video_path_params(video_path: str) -> Tuple[int, str, int, int]:
    """Parses params in video_path."""
    path = Path(video_path)
    y_ix, image_name, crop_fraction, video_len = path.stem.split("-")
    return int(y_ix), image_name, int(crop_fraction), int(video_len)


def rank(x: torch.Tensor, ix: int):
    """Gets rank of a (X, ) tensor."""
    return (torch.argsort(x, descending=True) == ix).nonzero().item()


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Computes the normalized entropy of a (B, X) tensor."""
    raw_entropy = torch.distributions.Categorical(logits=logits).entropy()
    normalized_entropy = raw_entropy / math.log(logits.shape[1])
    return normalized_entropy


def sharpen(p: torch.Tensor, T, dim: int) -> torch.Tensor:
    """Increase temperature of distribution."""
    p_T = torch.pow(p, 1 / T)
    return p_T / torch.sum(p_T, dim=dim, keepdim=True)
