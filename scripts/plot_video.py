import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FT
from torchvision.io import read_video

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})

video_path = "/Users/dgcnz/Downloads/4-50/979-valley-4-40.mp4"

vid = read_video(video_path, output_format="TCHW")[0]

R, C = 3, 3
fig, axes = plt.subplots(
    R,
    C,
    gridspec_kw=dict(
        wspace=0.05,
        hspace=0.05,
        top=1.0 - 0.5 / (R + 1),
        bottom=0.5 / (R + 1),
        left=0.5 / (C + 1),
        right=1 - 0.5 / (C + 1),
    ),
    figsize=(C + 1, R + 1),
    sharey="row",
    sharex="col",
)
labels = {
    0: "alp",
    5: "volcano",
    9: "alp",
    10: "snowmobile",
    15: "alp",
    19: "valley",
    21: "valley",
    24: "valley",
    29: "valley",
}
frames_ix = [
    [0, 5, 9],
    [10, 15, 19],
    [21, 24, 29],
]

for i in range(R):
    for j in range(C):
        ix = frames_ix[i][j]
        txt = f"({ix}) {labels[ix]}"
        frame = vid[ix]
        frame = FT.resize(frame, [256, 256])
        axes[i][j].imshow(frame.permute(1, 2, 0))
        axes[i][j].text(x=10, y=40, s=txt, fontsize=8)
        axes[i][j].axis("off")
# plt.show()
plt.savefig("scripts/figures/valley.png", bbox_inches="tight", dpi=400)
