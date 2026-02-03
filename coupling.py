import numpy as np

def deposit_sources(agents, grid_shape, dx, sigma):
    source = np.zeros(grid_shape)
    for pos, active in zip(agents.pos, agents.active):
        if not active:
            continue
        i = int(pos[0] / dx)
        j = int(pos[1] / dx)
        source[i % grid_shape[0], j % grid_shape[1]] += sigma
    return source
