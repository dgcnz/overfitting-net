import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as FT
from torch import Tensor


def zigzag(img: Tensor, hcrop, wcrop, maxlen=100) -> Tensor:
    """
    output (T, C, H, W)
    """
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

    return torch.stack(vid)


def display_video(vid: Tensor):
    fig, ax = plt.subplots()
    frames = []
    for ix, img in enumerate(vid):
        frame = ax.imshow(img.permute(1, 2, 0), animated=True)
        if ix == 0:
            ax.imshow(img.permute(1, 2, 0))
        frames.append([frame])
    return animation.ArtistAnimation(
        fig,
        frames,
        interval=50,
        blit=True,
        repeat_delay=1000,
    )


def float32_to_uint8(tensor: Tensor) -> Tensor:
    return (255 * tensor).type(torch.uint8)


def uint8_to_float32(tensor: Tensor) -> Tensor:
    return (tensor / 255.0).type(torch.float32)
