import matplotlib.pyplot as plt
from overfit.utils.misc import entropy
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch
import matplotlib.animation as animation
import matplotlib
from overfit.trainers.overfit import OverfitTrainer
from overfit.utils.misc import parse_video_path_params
from torchvision.io import read_video
from overfit.utils.misc import get_source_model

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})

VIDEO_PATH = "data/video/4-50/979-valley-4-40.mp4"
CONFIDENCE = 0.1
WEIGHT_DECAY = 0.4
MAX_LR = 0.4
MOMENTUM = 0.1


class ToFloat(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return (tensor / 255.0).type(torch.float32)


TRANSFORM_IMG = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        ToFloat(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cpu")
vid = read_video(VIDEO_PATH, output_format="TCHW")[0]
raw_vid = transforms.Compose([transforms.Resize((224, 224))])(torch.clone(vid))
vid = TRANSFORM_IMG(vid).to(device)
y_ix, _, crop_fraction, n_frames = parse_video_path_params(VIDEO_PATH)
srcnet = get_source_model("vit", device=device)

with open("imagenet_classes.txt", "r") as f:
    categories = f.readlines()
    categories = [cat.rstrip("\n") for cat in categories]

tgtnet_trainer = OverfitTrainer(categories=categories)
tgtnet_trainer.set(
    pretrained_classifier=srcnet,
    num_classes=1000,
    confidence=CONFIDENCE,
    weight_decay=WEIGHT_DECAY,
    max_lr=MAX_LR,
    momentum=MOMENTUM,
)
tgtnet_trainer.model = tgtnet_trainer.model.to(device)
frames = [frame.permute(1, 2, 0) for frame in raw_vid]
primes = []
ys_tgt = []
ys_src = []
certainties = []
for x in vid:
    y_tgt = tgtnet_trainer.forward_backward(x.unsqueeze(0))
    y_src = srcnet(x.unsqueeze(0))
    H_src = entropy(logits=y_src)[0].item()
    certainties.append(1 - float(H_src))
    ys_tgt.append(torch.clone(F.softmax(y_tgt, dim=1)).detach().numpy()[0])
    ys_src.append(torch.clone(F.softmax(y_src, dim=1)).detach().numpy()[0])
    prime = torch.clone(tgtnet_trainer.model.prime)
    primes.append(prime.detach().numpy())


light_blue = "#a8dadc"
mid_blue = "#457b9d"
red = "#e63946"
dark_blue = "#1d3557"

fig = plt.figure(tight_layout=True)
R = 2
C = 3
gs = gridspec.GridSpec(R, C)

ax_y = fig.add_subplot(gs[0, 1:])
ax_h = fig.add_subplot(gs[1, 1:])
ax_img = fig.add_subplot(gs[0, 0])
ax_prime = fig.add_subplot(gs[1, 0])

# indices = [str(i) for i in range(len(categories))]
best_categories = ["snowmobile", "dogsled", "volcano", "alp", "valley"]
best_indices = [categories.index(cat) for cat in best_categories]
map_cat_ix = dict(zip(best_categories, best_indices))

indices = best_categories
y_tgt_valley = [y[map_cat_ix["valley"]] for y in ys_tgt]
y_src_valley = [y[map_cat_ix["valley"]] for y in ys_src]
y_src_max = [max(y) for y in ys_src]


def subset(p):
    return np.take(p, best_indices)


prime_plot = ax_prime.bar(x=indices, height=subset(primes[0]), color=red, label="$P_i$")
ax_prime.set_ylim(-0.5, 3)
ax_h.set_ylim(0, 1)
ax_h.set_xlim(0, len(primes))
ax_h.set_xticks(list(range(len(primes)))[::5])
ax_y.set_ylim(0, 1)
ax_y.set_xlim(0, len(primes))
ax_y.set_xticks(list(range(len(primes)))[::5])
ax_img.imshow(frames[0])
ax_img.set_anchor("W")
# y_plot = ax_y.bar(x=indices, height=subset(ys[0]))
# ax_y.scatter(x=range(len(ys)), y=[])

ax_y.scatter(
    x=[],
    y=[],
    color=light_blue,
    marker=".",
    label="$M_T(x_i)_{{valley}}$",
)

ax_y.scatter(
    x=[],
    y=[],
    color=dark_blue,
    marker=".",
    label="$M_S(x_i)_{{valley}}$",
)

# ax_y.scatter(
#     x=[],
#     y=[],
#     color=mid_blue,
#     marker=".",
#     label="\\max($M_S(x_i))$",
# )

ax_h.scatter(
    x=[],
    y=[],
    color=red,
    marker=".",
    label="$1 - \\bar{H}(M_S(x_i))$",
)
ax_prime.set_xticks(best_categories, rotation=45, ha="right")
ax_prime.set_xticklabels(best_categories, rotation=45)

legend = fig.legend(loc="center right", fancybox=True)  # Define legend objects
# legend = fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


def animate(i):
    p = subset(primes[i])
    # y = subset(ys[i])

    for j, b in enumerate(prime_plot):
        b.set_height(p[j])

    xx = list(range(i + 1))
    ax_prime_label = "$P_{" + str(i) + "}$"
    ax_y_tgt_label = "$M_T(x_{" + str(i) + "})_{{valley}}$"
    ax_y_src_label = "$M_S(x_{" + str(i) + "})_{{valley}}$"
    # ax_y_max_label = "$\\max(M_S(x_{" + str(i) + "}))$"
    ax_h_label = "$1 - \\bar{H}(M_S(x_{" + str(i) + "}))$"

    legend.get_texts()[2].set_text(ax_h_label)
    legend.get_texts()[0].set_text(ax_y_tgt_label)
    legend.get_texts()[1].set_text(ax_y_src_label)
    # legend.get_texts()[2].set_text(ax_y_max_label)
    legend.get_texts()[3].set_text(ax_prime_label)
    ax_y.scatter(
        x=xx,
        y=y_tgt_valley[: (i + 1)],
        color=light_blue,
        marker=".",
    )
    ax_y.scatter(
        x=xx,
        y=y_src_valley[: (i + 1)],
        color=dark_blue,
        marker=".",
    )

    # ax_y.scatter(
    #     x=xx,
    #     y=y_src_max[: (i + 1)],
    #     color=mid_blue,
    #     marker=".",
    # )
    ax_h.scatter(
        x=xx,
        y=certainties[: (i + 1)],
        color=red,
        marker=".",
    )

    # for j, b in enumerate(y_plot):
    #     b.set_height(y[j])
    ax_img.imshow(frames[i], animated=True)


# fig.set_size_inches(14, 8, True)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(primes),
    interval=200,
    repeat=True,
    blit=False,
    save_count=50,
)
ani.save("scripts/figures/animation.mp4", bitrate=5000, dpi=500)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

# plt.show()
