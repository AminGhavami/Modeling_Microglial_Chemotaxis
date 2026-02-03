# Domain
Lx, Ly = 10.0, 10.0
Nx, Ny = 100, 100
dx = Lx / Nx
dy = Ly / Ny
dt = 0.01

# PDE parameters — tuned for strong relay amplification
D = 0.25          # faster diffusion
gamma = 0.03      # slow decay → signal can build up
sigma = 4.0       # each active cell adds a lot

# Agent parameters
chi = 0.8
D_eff = 0.04
threshold = 0.6   # ← only agents very close to the hot spot activate first
epsilon = 1e-6

# Simulation
n_agents = 300    # ← crucial: more agents = visible recruitment wave
steps = 3000