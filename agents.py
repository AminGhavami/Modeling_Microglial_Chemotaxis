import numpy as np

class Agents:
    def __init__(self, n, Lx, Ly):
        self.pos = np.random.rand(n, 2) * [Lx, Ly]
        self.active = np.zeros(n, dtype=bool)

    def update_state(self, c, threshold, dx):
        ix = np.clip((self.pos[:, 0] / dx).astype(int), 0, c.shape[0]-1)
        iy = np.clip((self.pos[:, 1] / dx).astype(int), 0, c.shape[1]-1)

        newly_active = c[ix, iy] > threshold
        self.active = np.logical_or(self.active, newly_active)

    
    def move(self, grad_c, chi, D_eff, dt, epsilon, Lx, Ly):
        drift = chi * grad_c / (np.linalg.norm(grad_c, axis=1, keepdims=True) + epsilon)
        noise = np.sqrt(2 * D_eff * dt) * np.random.randn(*self.pos.shape)
        self.pos += dt * drift + noise
        self.pos[:, 0] %= Lx
        self.pos[:, 1] %= Ly
