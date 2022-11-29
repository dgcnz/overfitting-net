import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})


wds = [0.0, 0.2, 0.4]
fig, axes = plt.subplots(1, 3)
light_blue = "#a8dadc"
mid_blue = "#457b9d"
red = "#e63946"
dark_blue = "#1d3557"

for ax, wd, ix in zip(axes, wds, range(len(wds))):
    df = pd.read_csv(f"scripts/data/wd{wd}_prime.csv")
    correct_df = df[df["key"] == "Correct prime"].sort_values(by="step", inplace=False)
    max_df = df[df["key"] == "Prime Max"].sort_values(by="step", inplace=False)

    ax.set_ylim(0, 3)
    ax.set_xlabel("Frame Index")
    ax.set_title(f"$\lambda={wd}$")
    if ix == 0:
        ax.plot(
            correct_df["step"],
            correct_df["value"],
            label=r"$P_{valley}$",
            marker=".",
            c=mid_blue,
        )
        ax.plot(max_df["step"], max_df["value"], label="$\max{P}$", marker=".", c=red)
    else:
        ax.plot(correct_df["step"], correct_df["value"], marker=".", c=mid_blue)
        ax.plot(max_df["step"], max_df["value"], marker=".", c=red)
fig.legend()
plt.savefig("scripts/figures/wdvalley_prime.png", dpi=400)
