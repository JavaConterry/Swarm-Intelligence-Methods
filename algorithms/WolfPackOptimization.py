import random as rand
import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

max_iter = 400   # iteration limit
s = 50           # number of wolves
step = 0.1       # speed of wolves

# Initialize wolves' locations
wolfs_locations = np.array([[rand.randint(-100, 100)/50, rand.randint(-100, 100)/50] for _ in range(s)])

def optimisation_function(coord):
    x, y = coord[0], coord[1]
    return x**2 + 2*y**2 - 0.3 * m.cos(3*m.pi*x) - 0.4 * m.cos(3*m.pi*y) + 0.7

def update_GBest(locs):
    fs = [optimisation_function(loc) for loc in locs]
    return locs[np.argmin(fs)]

GBest = update_GBest(wolfs_locations)

fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
sc = ax.scatter(wolfs_locations[:, 0], wolfs_locations[:, 1], c='blue', label='Wolves')
GBest_sc = ax.scatter(GBest[0], GBest[1], c='red', marker='x', label='Global Best')
ax.legend()

def update(frame):
    global GBest, wolfs_locations

    
    for wolf, wolf_loc in enumerate(wolfs_locations):
        if np.linalg.norm(GBest - wolf_loc) > 1e-6:
            wolfs_locations[wolf] = wolf_loc + step * (GBest - wolf_loc) / np.linalg.norm(GBest - wolf_loc)
    GBest = update_GBest(wolfs_locations)
    
    sc.set_offsets(wolfs_locations)
    GBest_sc.set_offsets(GBest)
    return sc, GBest_sc

ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=100, repeat=False)
plt.show()

print(f'min point: {GBest}; min of function {optimisation_function(GBest)}')