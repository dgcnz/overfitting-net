import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})


momentums = [0.1, 0.4]
fig, axes = plt.subplots(1, 2)
light_blue = "#a8dadc"
mid_blue = "#457b9d"
red = "#e63946"
dark_blue = "#1d3557"

for ax, momentum, ix in zip(axes, momentums, range(len(momentums))):
    df = pd.read_csv(f"scripts/data/m{momentum}_prime.csv")
    correct_df = df[df["key"] == "Correct prime"].sort_values(by="step", inplace=False)
    max_df = df[df["key"] == "Prime Max"].sort_values(by="step", inplace=False)

    ax.set_ylim(0, 2.1)
    ax.set_xlabel("Frame Index")
    ax.set_title(f"$\\beta={momentum}$")
    if ix == 0:
        ax.plot(
            correct_df["step"],
            correct_df["value"],
            label=r"$P_{snail}$",
            marker=".",
            c=mid_blue,
        )
        ax.plot(max_df["step"], max_df["value"], label="$\max{P}$", marker=".", c=red)
    else:
        ax.plot(correct_df["step"], correct_df["value"], marker=".", c=mid_blue)
        ax.plot(max_df["step"], max_df["value"], marker=".", c=red)
fig.legend()
plt.savefig("scripts/figures/msnail_prime.png", dpi=400)
