import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.io import read_video
import itertools

x = ["mop", "komondor"]
y_mop = [0.7, 0.1, 0.01, 0.5, 0.5, 0.7]
y_komondor = [0.3, 0.9, 0.99, 0.5, 0.5, 0.3]
y = list(zip(y_mop, y_komondor))


def avg(prv, cur):
    y0 = prv[0] + cur[0]
    y1 = prv[1] + cur[1]
    sm = y0 + y1
    return (y0 / sm, y1 / sm)


y_avg = list(itertools.accumulate(y, avg))
print(y_avg)


video, *_ = read_video("data/video/228-komondormop-2-6.mp4", output_format="TCHW")

fig, axes = plt.subplots(2, 6)

komondor_color = "#457b9d"
mop_color = "#1d3557"
avg_color = "#e63946"
for i, frame in enumerate(video):
    axes[0][i].imshow(frame.permute(1, 2, 0))
    axes[0][i].axis("off")
    last_komondor = axes[1][i].bar(
        [0], y_komondor[i], bottom=y_mop[i], color=komondor_color
    )
    last_mop = axes[1][i].bar([0], y_mop[i], color=mop_color)
    # axes[1][i].get_yaxis().set_visible(False)
    axes[1][i].axis("off")
    last_mid = axes[1][i].axhline(y=0.5, color="white", linestyle="dashed")
    last_avg = axes[1][i].axhline(y=y_avg[i][0], color=avg_color)
    axes[1][i].set_ylim(0, 1)

handles = [
    mpatches.Patch(color=komondor_color, label="P(Komondor)"),
    mpatches.Patch(color=mop_color, label="P(Mop)"),
    mpatches.Patch(color=avg_color, label="Running average"),
]
fig.legend(handles=handles)

plt.show()
