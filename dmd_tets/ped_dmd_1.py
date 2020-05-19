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
from pydmd import DMD
from past.utils import old_div
import matplotlib.pyplot as plt
from matplotlib import animation


def distance_field(grid, state):
    start = time.time()
    grid_x = grid[:,0]
    grid_y = grid[:,1]
    state_x = state[:,0][:, np.newaxis]
    state_y = state[:,1][:, np.newaxis]
    diff_x = grid_x - state_x
    diff_y = grid_y - state_y
    diff_xy = np.sqrt(diff_x**2 + diff_y**2)
    dist_xy = diff_xy.min(axis=0)
    # print("time: ", time.time()-start)
    return dist_xy

def animate2(states, space, dest=None):
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
        vals = distance_field(grid, snapshot)
        vals = vals.reshape(50,50)
        ax2.cla()
        ax2.contourf(*xy, vals, levels=50)

        return [peds]
    # start animation
    anim = animation.FuncAnimation(fig, sub_animate, frames=time, interval=40)
    plt.show()


def data_gen():
    field_size = 15
    num_peds = 40
    dest = np.array([
            [2.0, 2.0],
            [2.0, 7.5],
            [2.0, 13.0],
            [7.5, 2.0],
            [7.5, 7.5],
            [7.5, 13.0],
            [13.0, 2.0],
            [13.0, 7.5],
            [13.0, 13.0]
        ])

    initial_state = []
    for i in range(num_peds):
        ped = np.zeros(6)
        ped[0] = np.random.uniform(1, 14)
        ped[1] = np.random.uniform(1, 14)
        ped[2] = np.random.uniform(0.3, 0.7)
        ped[3] = np.random.uniform(0.3, 0.7)
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
    states = np.stack([s.step().state.copy() for _ in range(720)])

    # print(space[0].shape)

    # animate2(states, space, dest)
    return states

if __name__ == "__main__":
    states = data_gen()[:, :, 0:4]
    print("test data generated.")
    states = np.reshape(states, (states.shape[0], -1)).T
    train_data = states[:,500:701]
    test_data = states[:,700:]

    dmd = DMD(svd_rank=40)
    dmd.fit(train_data)
    print("DMD fit finished.")

    omega = old_div(np.log(dmd.eigs), dmd.original_time['dt'])
    dmd_timesteps = np.arange(200, 220, 1)
    vander = np.exp(np.outer( omega,  dmd_timesteps - dmd.original_time['t0'] ))
    dynamics = vander * dmd._b[:, None]
    test_data_dmd = np.dot(dmd.modes, dynamics)
    print("DMD prediction finished.")

    print(test_data.shape)
    print((test_data_dmd[1,:] - test_data[1,:]).real)


    print(train_data[0:10,-1])
    print(test_data[0:10, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')

    snapshot = train_data[:, -1].reshape(40, -1)
    ped_xy = snapshot[:,0:2].T
    ax.scatter(ped_xy[0], ped_xy[1], c='r', s=100)

    diff = (test_data_dmd[:,0] - test_data[:,0]).real
    print("diff.shape: ", diff.shape)
    for i in range(diff.shape[0]):
        test_data_dmd[i,:] -= diff[i]
    test_data_dmd = test_data_dmd.real

    print("verify first element truncation: ", test_data_dmd[0:10,0], test_data[0:10,0])

    print(test_data.shape)
    for p in range(40):
        traj = test_data[4*p : 4*p+2, :]
        ax.scatter(traj[0,:], traj[1,:], c='b', s=5)

        traj_dmd = test_data_dmd[4*p : 4*p+2, :]
        ax.scatter(traj_dmd[0,:], traj_dmd[1,:], c='y', s=5)
    plt.show()
