import urllib
from re import A
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torch import Tensor
from torchvision import transforms


def zigzag(img: Tensor, hcrop, wcrop, maxlen=100) -> List[Tensor]:
    _, h, w = img.size()
    k = h // hcrop
    wstep = w // (maxlen // k)
    hstep = hcrop
    vid = []
    for ix, top in enumerate(range(0, h - hcrop + 1, hstep)):
        row_vid = []
        for left in range(0, w - wcrop + 1, wstep):
            crop = FT.crop(img, top, left, hcrop, wcrop)
            row_vid.append(crop)
        if ix % 2 == 1:
            row_vid.reverse()
        vid.extend(row_vid)

    return vid


def display_video(vid: List[Tensor]):
    fig, ax = plt.subplots()
    frames = []
    for ix, img in enumerate(vid):
        frame = ax.imshow(img.permute(1, 2, 0), animated=True)
        if ix == 0:
            ax.imshow(img.permute(1, 2, 0))
        frames.append([frame])
    _ = animation.ArtistAnimation(
        fig,
        frames,
        interval=50,
        blit=True,
        repeat_delay=1000,
    )
    plt.show()
