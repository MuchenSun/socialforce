"""
This version aims for animation and sliding window mechanism
it's the second version, try to optimize the parameters
"""

import sys
sys.path.append('/home/msun/Code/socialforce')

from contextlib import contextmanager
import numpy as np
import pytest
import socialforce
import time
from pydmd import HODMD, DMD
from past.utils import old_div
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

terminal = 1000
window_size = 200
predict_size = 5

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

def animate2(states, space, dest=None, true_traj=None, dmd_traj=None):
    # render data to get necessary parameters
    time = states.shape[0]
    num_ped = states.shape[1]
    # configure canvas
    fig = plt.gcf()
    # other configuration
    true_traj = np.array(true_traj).real
    dmd_traj = np.array(dmd_traj).real
    print("true_traj.shape:", true_traj.shape)
    print("dmd_traj.shape: ", dmd_traj.shape)

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
    true_traj_points = ax1.scatter([], [], c='b', s=5)
    dmd_traj_points = ax1.scatter([], [], c='y', s=5)

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

        # Figure 1: traj
        if t < window_size-1 or t > terminal-predict_size-1:
            true_traj_points.set_offsets([-10, -10])
            dmd_traj_points.set_offsets([-10, -10])
        else:
            curr_true_traj_x = true_traj[t+1-window_size][0::2].reshape(-1)
            curr_true_traj_y = true_traj[t+1-window_size][1::2].reshape(-1)
            true_traj_points.set_offsets(np.array([curr_true_traj_x, curr_true_traj_y]).T)

            curr_dmd_traj_x = dmd_traj[t+1-window_size][0::2].reshape(-1)
            curr_dmd_traj_y = dmd_traj[t+1-window_size][1::2].reshape(-1)
            dmd_traj_points.set_offsets(np.array([curr_dmd_traj_x, curr_dmd_traj_y]).T)

        # Figure 2: distance field
        ax2.clear()
        vals = distance_field(grid, snapshot)
        vals = vals.reshape(50,50)
        ax2.cla()
        ax2.contourf(*xy, vals, levels=50)

        return [peds, true_traj_points, dmd_traj_points]
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
    # states = np.stack([s.step().state.copy() for _ in range(1020)])

    # main simultion loop
    states = []
    true_traj = []
    dmd_traj = []

    dmd = DMD(svd_rank=40, opt=True)

    counter = 0
    for i in tqdm(range(terminal)):
        curr_state = s.step().state.copy()
        states.append(curr_state)
        if i < window_size-1 or i > terminal-predict_size-1:
            pass
        else:
            counter += 1
            window = np.array( states[i+1-window_size:i+1] )
            train_data = np.reshape( window[:, :, 0:2],
                                     (window.shape[0], -1) ).T
            dmd = HODMD(svd_rank=40)
            dmd.fit(train_data)
            # print(i, train_data.shape)

            omega = old_div(np.log(dmd.eigs), dmd.original_time['dt'])
            dmd_timesteps = np.arange(i, i+predict_size, 1)
            vander = np.exp(
                    np.outer(omega, dmd_timesteps-0)
                )
            dynamics = vander * dmd._b[:, None]
            predict_data = np.dot(dmd.modes, dynamics)
            diff = predict_data[:,0] - curr_state[:,0:2].reshape(-1)
            predict_data -= diff[:,np.newaxis]
            dmd_traj.append(predict_data)
    print("counter: ", counter)

    for i in np.arange(window_size-1+predict_size, terminal, 1):
        traj = np.array( states[i-predict_size:i] )
        test_data = np.reshape( traj[:, :, 0:2],
                                (traj.shape[0], -1) ).T

        true_traj.append(test_data)


    states = np.array(states)

    # print(space[0].shape)

    animate2(states, space, dest, true_traj, dmd_traj)
    return states

if __name__ == "__main__":
    states = data_gen()[:, :, 0:2]

