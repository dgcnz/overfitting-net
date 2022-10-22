from pathlib import Path
from typing import Tuple


def floor_even(x: int) -> int:
    return 2 * (x // 2)


def parse_video_filename_params(video_path: str) -> Tuple[int, int]:
    crop_fraction, video_len = Path(video_path).stem.split("-")
    return int(crop_fraction), int(video_len)
