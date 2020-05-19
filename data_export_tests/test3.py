"""
This test is for double animation with pedestrain and distance field
"""

import sys
sys.path.append('/home/msun/Code/socialforce')

from contextlib import contextmanager
import numpy as np
import pytest
import socialforce
import time

def distance_field(grid, state):
    """compute distance field"""
    vals = np.zeros(grid.shape[0])
    for i in range(grid.shape[0]):
        cell = grid[i,:]
        dists = []
        for j in range(state.shape[0]):
            agent = state[j][0:2]
            dist = np.sqrt( np.sum((cell-agent)**2) )
            vals[i] += dist
    vals = vals / np.sum(vals)
    return vals

def distance_field2(grid, state):
    vals = np.zeros(grid.shape[0])
    for i in range(state.shape[0]):
        vals += np.sqrt( np.sum( (grid-state[i][0:2])**2 , axis=1) )
    return vals / np.sum(vals)

def distance_field3(grid, state):
    start = time.time()
    vals = np.zeros(grid.shape[0])
    for i in range(grid.shape[0]):
        vals[i] += np.sqrt( np.sum( (grid[i,:]-state[:,0:2])**2 , axis=1) ).min()
    print("time: ", time.time()-start)
    return vals / np.sum(vals)

def distance_field4(grid, state):
    start = time.time()
    grid_x = grid[:,0]
    grid_y = grid[:,1]
    state_x = state[:,0][:, np.newaxis]
    state_y = state[:,1][:, np.newaxis]
    diff_x = grid_x - state_x
    diff_y = grid_y - state_y
    diff_xy = np.sqrt(diff_x**2 + diff_y**2)
    dist_xy = diff_xy.min(axis=0)
    print("time: ", time.time()-start)
    return dist_xy

def animate2(states, space, dest=None):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # render data to get necessary parameters
    time = states.shape[0]
    num_ped = states.shape[1]
    # configure canvas
    fig = plt.gcf()

    # Figure 1: pedestrian visualization
    ax1 = fig.add_subplot(121)
    # generate background (plot space)
    for item in space: # each item is a 2D array
        ax1.scatter(item[:,0], item[:,1], c='k')
    if dest is not None:
        ax1.scatter(dest[:,0], dest[:,1], c='b', marker='+', s=100)
    # configure canvas
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(-0.5, 15.5)
    # initialize plots for peds
    peds = ax1.scatter([], [], c='r')

    # Figure 2: distance field visualization
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal', 'box')
    grid = np.meshgrid(*[np.linspace(0, 15, 50) for _ in range(2)])
    grid = np.c_[grid[0].ravel(), grid[1].ravel()] # 2*N array
    xy = []
    for g in grid.T:
        xy.append(np.reshape(g, newshape=(50,50)))

    # define subfunction for animation (this is for all subplots)
    def sub_animate(t): # i indicates time step
        # Figure 1: pedestrian visualization
        # get current system snapshot ...
        snapshot = states[t] # ... which should be a 2D matrix
        # extract x and y positions
        x_pos = snapshot[:,0]
        y_pos = snapshot[:,1]
        # scatter
        peds.set_offsets(snapshot[:,0:2])

        # Figure 2: distance field
        ax2.clear()
        vals = distance_field4(grid, snapshot)
        vals = vals.reshape(50,50)
        ax2.cla()
        ax2.contourf(*xy, vals, levels=50)

        return [peds]
    # start animation
    anim = animation.FuncAnimation(fig, sub_animate, frames=time, interval=40)
    plt.show()


def main():
    field_size = 15
    num_peds = 80
    dest = np.array([
            [2.0, 2.0],
            [2.0, 13.0],
            [7.5, 7.5],
            [13.0, 2.0],
            [13.0, 13.0]
        ])

    initial_state = []
    for i in range(num_peds):
        ped = np.zeros(6)
        ped[0] = np.random.uniform(1, 5)
        ped[1] = np.random.uniform(1, 5)
        ped[2] = np.random.uniform(0.3, 0.8)
        ped[3] = np.random.uniform(0.3, 0.8)
        dest_id = np.random.randint(0, len(dest))
        ped[4] = dest[dest_id][0]
        ped[5] = dest[dest_id][1]
        initial_state.append(ped)
    initial_state = np.array(initial_state)

    space = [
        np.array([(x, 0) for x in np.linspace(0, 15, 1000)]),
        np.array([(x,15) for x in np.linspace(0, 15, 1000)]),
        np.array([( 0, y) for y in np.linspace(0, 15, 1000)]),
        np.array([(15, y) for y in np.linspace(0, 15, 1000)])
    ]
    ped_space = socialforce.PedSpacePotential(space)

    s = socialforce.SimulatorMultiDest(initial_state=initial_state,
                                       ped_space=ped_space,
                                       dest=dest,
                                       delta_t=0.1)
    states = np.stack([s.step().state.copy() for _ in range(1200)])

    print(space[0].shape)

    animate2(states, space, dest)

if __name__ == "__main__":
    main()
