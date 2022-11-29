import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman, Times"]})

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_aspect("equal")

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
# for i in range(2):
#    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)
elev = 10.0
rot = 80.0 / 180 * np.pi
ax.plot_surface(
    x, y, z, rstride=4, cstride=4, color="#e63946a0", linewidth=0, alpha=0.5
)
# calculate vectors for "vertical" circle
a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
b = np.array([0, 1, 0])
b = (
    b * np.cos(rot)
    + np.cross(a, b) * np.sin(rot)
    + a * np.dot(a, b) * (1 - np.cos(rot))
)
ax.plot(np.sin(u), np.cos(u), 0, color="#e63946", linestyle="dashed")
horiz_front = np.linspace(0, np.pi, 100)
ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color="#e63946")
vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
ax.plot(
    a[0] * np.sin(u) + b[0] * np.cos(u),
    b[1] * np.cos(u),
    a[2] * np.sin(u) + b[2] * np.cos(u),
    color="#e63946",
    linestyle="dashed",
)
ax.plot(
    a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    b[1] * np.cos(vert_front),
    a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),
    color="#e63946",
)
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(elev=elev, azim=0)


plt.savefig("sphere.png", dpi=300)
plt.show()
