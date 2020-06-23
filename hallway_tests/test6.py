"""
This test is based on test3, for testing different pedestrian
motion with distance field
"""

import sys
sys.path.append('/home/msun/Code/socialforce')

from contextlib import contextmanager
import numpy as np
import pytest
import socialforce
import time


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

def distance_field5(grid, state, space):
    start = time.time()
    grid_x = grid[:,0]
    grid_y = grid[:,1]
    space = np.array(space).reshape(-1,2)
    space_x = space[:,0]
    space_y = space[:,1]
    print(space_x.shape)
    state_x = np.concatenate((state[:,0], space_x))[:, np.newaxis]
    state_y = np.concatenate((state[:,1], space_y))[:, np.newaxis]
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
    ax1 = fig.add_subplot(211)
    # generate background (plot space)
    for item in space: # each item is a 2D array
        ax1.scatter(item[:,0], item[:,1], c='k')
    if dest is not None:
        ax1.scatter(dest[:,0], dest[:,1], c='b', marker='+', s=100)
    # configure canvas
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5,  2.5)
    # initialize plots for peds
    peds = ax1.scatter([], [], c='r')

    # Figure 2: distance field visualization
    ax2 = fig.add_subplot(212)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5 ,2.5)
    grid = np.meshgrid(*[np.linspace(0, 10, 50), np.linspace(0, 2, 50)])
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
        vals = distance_field5(grid, snapshot, space)
        vals = vals.reshape(50,50)
        ax2.cla()
        ax2.contourf(*xy, vals, levels=50)

        return [peds]
    # start animation
    anim = animation.FuncAnimation(fig, sub_animate, frames=time, interval=25)
    plt.show()


def main():
    field_size = 15
    num_peds = 2
    dest = np.array([
            [1.0, 1.0],
            [9.0, 1.0]
        ])

    initial_state = []
    for i in range(num_peds):
        ped = np.zeros(6)
        ped[0] = np.random.uniform(1, 9)
        ped[1] = np.random.uniform(0.2, 1.8)
        ped[2] = np.random.uniform(0.3, 0.7)
        ped[3] = np.random.uniform(0.3, 0.7)
        dest_id = np.random.randint(0, len(dest))
        ped[4] = dest[dest_id][0]
        ped[5] = dest[dest_id][1]
        initial_state.append(ped)
    initial_state = np.array(initial_state)

    space = [
        np.array([(x, 0) for x in np.linspace(0, 10, 200)]),
        np.array([(x, 2) for x in np.linspace(0, 10, 200)]),
        np.array([( 0, y) for y in np.linspace(0, 2, 200)]),
        np.array([(10, y) for y in np.linspace(0, 2, 200)])
    ]
    ped_space = socialforce.PedSpacePotential(space)

    s = socialforce.SimulatorMultiDest(initial_state=initial_state,
                                       ped_space=ped_space,
                                       dest=dest,
                                       delta_t=0.1)
    states = np.stack([s.step().state.copy() for _ in range(1000)])

    print(space[0].shape)

    animate2(states, space, dest)

if __name__ == "__main__":
    main()
