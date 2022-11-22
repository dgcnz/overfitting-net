import math
from pathlib import Path
from typing import Tuple

import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
)


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


def batch(iterable, n=1):
    """Generate batches of maximum size `n`"""
    sz = len(iterable)
    for ndx in range(0, sz, n):
        mx = min(ndx + n, sz)
        yield iterable[ndx:mx]


def get_source_model(model: str, device):
    if model == "vit":
        return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(device)
    elif model == "resnet34":
        return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval().to(device)
    elif model == "resnet50":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    elif model == "resnet18":
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    else:
        raise Exception("Unknown Source model")
