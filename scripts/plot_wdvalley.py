import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})

df = pd.read_csv("scripts/data/wdvalley.csv")

fig, ax = plt.subplots(1, 1)

marker = "."
ax.set_ylabel("Accumulated Top-1 Count")
ax.set_xlabel("Frame Index")
ax.plot(df["step"], df["source"], c="#e63946", marker=marker, label="$M_S$")
ax.plot(df["step"], df["0.0"], c="#a8dadc", marker=marker, label="$M_T(\lambda=0.0)$")
ax.plot(df["step"], df["0.2"], c="#457b9d", marker=marker, label="$M_T(\lambda=0.2)$")
ax.plot(df["step"], df["0.4"], c="#1d3557", marker=marker, label="$M_T(\lambda=0.4)$")
fig.legend()
plt.savefig("scripts/figures/wdvalley_plot.png", dpi=400)
