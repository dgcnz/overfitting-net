import argparse
import logging
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as FT
from overfit.utils.img2vid import zigzag
from PIL import Image
from torchvision.io.video import write_video

VIDEO_OUT_PATH = Path(os.getenv("VIDEO_OUT_PATH", "data/videos"))
logging.basicConfig(level=logging.DEBUG)


def generate_video(img_path: str, crop_fraction: int, max_length: int) -> str:
    logging.info("Creating video")
    input_image = Image.open(img_path)
    input_tensor = FT.to_tensor(input_image)
    _, h, w = input_tensor.size()

    crop_fraction = crop_fraction
    hcrop = h // crop_fraction
    wcrop = w // crop_fraction
    hcrop = (hcrop // 2) * 2
    wcrop = (wcrop // 2) * 2
    input_video = zigzag(input_tensor, hcrop, wcrop, max_length)
    video_tensor = torch.stack(input_video)
    video_tensor = video_tensor.permute(0, 2, 3, 1).type(torch.uint8)
    logging.info(video_tensor.shape)
    logging.info(video_tensor.dtype)
    image_filename = Path(img_path).stem
    video_dir = VIDEO_OUT_PATH / image_filename
    video_dir.mkdir(parents=True, exist_ok=True)
    out_name = video_dir / f"{crop_fraction}-{len(input_video)}.mp4"
    write_video(filename=str(out_name), video_array=video_tensor, fps=1)
    return str(out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("img_path", type=str, help="Source image path.")
    parser.add_argument("--crop_fraction", type=int, help="Crop fraction", default=3)
    parser.add_argument("--max_length", type=int, help="Max length", default=100)
    args = parser.parse_args()
    out_name = generate_video(
        img_path=args.img_path,
        crop_fraction=args.crop_fraction,
        max_length=args.max_length,
    )
    logging.info(f"OUT FILE: {out_name}")
