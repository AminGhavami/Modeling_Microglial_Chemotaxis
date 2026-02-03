import matplotlib.pyplot as plt
from simulation import c, agents
import numpy as np

def compute_radial_profile(c):
    y, x = np.indices(c.shape)
    center = np.array(c.shape) // 2
    r = np.sqrt((x-center[1])**2 + (y-center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), c.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def plot_signal(c, title="Signal field"):
    plt.figure(figsize=(5,4))
    plt.imshow(c.T, origin="lower", cmap="inferno")
    plt.colorbar(label="Signal concentration")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6,5))
plt.imshow(c.T, origin="lower", cmap="inferno")
plt.colorbar(label="Signal concentration")
plt.scatter(
    agents.pos[:,0]/(10/100),
    agents.pos[:,1]/(10/100),
    c="cyan", s=10
)
plt.title("Microglial Signal Relay")
plt.show()
