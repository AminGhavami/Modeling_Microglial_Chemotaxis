import matplotlib.pyplot as plt
from simulation import c, agents

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
