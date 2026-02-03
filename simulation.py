import numpy as np
from parameters import *
from agents import Agents
from pde import update_signal
from coupling import deposit_sources
from visualize import plot_signal

# Initialize
c = np.zeros((Nx, Ny))
agents = Agents(n_agents, Lx, Ly)

# Small, strong initial injury (only ~20–40 agents will see >0.6 at t=0)
center_x = Nx // 2
center_y = Ny // 2
sigma_patches = 4.0
xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
c = 3.0 * np.exp(-dist**2 / (2 * sigma_patches**2))

# Tiny pre-diffusion (just smooths the blob a bit)
for _ in range(10):
    c = update_signal(c, np.zeros((Nx, Ny)), D, gamma, dt, dx, dy)

print("Initial max signal:", c.max())   # should be ~3.0

activated_counts = []
mean_distances = []

center_phys = np.array([Lx/2, Ly/2])

for step in range(steps):
    agents.update_state(c, threshold, dx)

    source = deposit_sources(agents, c.shape, dx, sigma)
    c = update_signal(c, source, D, gamma, dt, dx, dy)

    # gradient & move (your existing code)
    grad_x, grad_y = np.gradient(c, dx)
    ix = (agents.pos[:,0] / dx).astype(int) % Nx
    iy = (agents.pos[:,1] / dx).astype(int) % Ny
    grad = np.stack([grad_x[ix, iy], grad_y[ix, iy]], axis=1)
    agents.move(grad, chi, D_eff, dt, epsilon, Lx, Ly)

    # metrics
    activated_counts.append(np.sum(agents.active))
    distances = np.linalg.norm(agents.pos - center_phys, axis=1)
    mean_distances.append(distances.mean())

    # snapshots
    if step in [0, 300, 800, 1500, 2800]:
        plot_signal(c, title=f"Signal at t = {step*dt:.2f}")