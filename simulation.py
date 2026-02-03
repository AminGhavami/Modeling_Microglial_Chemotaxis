import numpy as np
from parameters import *
from agents import Agents
from pde import update_signal
from coupling import deposit_sources

# Initialize
c = np.zeros((Nx, Ny))
agents = Agents(n_agents, Lx, Ly)

# Initial injury signal
c[Nx//2, Ny//2] = 5.0

for step in range(steps):
    agents.update_state(c, threshold, dx)

    source = deposit_sources(agents, c.shape, dx, sigma)
    c = update_signal(c, source, D, gamma, dt, dx, dy)

    grad_x, grad_y = np.gradient(c, dx)
    ix = (agents.pos[:, 0] / dx).astype(int)
    iy = (agents.pos[:, 1] / dx).astype(int)
    grad = np.stack([grad_x[ix, iy], grad_y[ix, iy]], axis=1)

    agents.move(grad, chi, D_eff, dt, epsilon, Lx, Ly)

    if step % 200 == 0:
        print(f"Step {step}")
