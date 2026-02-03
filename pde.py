import numpy as np

def laplacian(c, dx, dy):
    return (
        np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
        np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) -
        4 * c
    ) / dx**2


def update_signal(c, source, D, gamma, dt, dx, dy):
    return c + dt * (D * laplacian(c, dx, dy) - gamma * c + source)
