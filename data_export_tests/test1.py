from contextlib import contextmanager
import numpy as np
import pytest
import socialforce


def animate(states, space, output_filename):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    # render data to get necessary parameters
    time = states.shape[0]
    num_ped = states.shape[1]
    # generate background (plot space)
    for item in space: # each item is a 2D array
        plt.scatter(item[:,0], item[:,1], c='k')
    # configure canvas
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2, 17)
    ax.set_ylim(-2, 17)
    # initialize plots for peds
    peds = ax.scatter([], [], c='r')
    # define subfunction for animation
    def sub_animate(t): # i indicates time step
        # get current system snapshot ...
        snapshot = states[t] # ... which should be a 2D matrix
        # extract x and y positions
        x_pos = snapshot[:,0]
        y_pos = snapshot[:,1]
        print('x_pos: ', x_pos)
        # scatter
        peds.set_offsets(snapshot[:,0:2])
        return [peds]
    # start animation
    anim = animation.FuncAnimation(fig, sub_animate, frames=time, interval=20, blit=True)
    plt.show()

def main():
    field_size = 15
    num_peds = 20
    dest = np.array([
            [2.0, 2.0],
            [2.0, 13.0],
            [13.0, 2.0],
            [13.0, 13.0]
        ])
    initial_state = []
    for i in range(num_peds):
        ped = np.zeros(6)
        ped[0] = np.random.uniform(0, field_size)
        ped[1] = np.random.uniform(0, field_size)
        ped[2] = np.random.uniform(0.3, 0.8)
        ped[3] = np.random.uniform(0.3, 0.8)
        dest_id = np.random.randint(0, len(dest))
        ped[4] = dest[dest_id][0]
        ped[5] = dest[dest_id][1]
        initial_state.append(ped)
    initial_state = np.array(initial_state)

    initial_state = np.array([
        [12.0,  2.0, 0.5, 0.0,  2.0, 2.0],
        [10.0, 10.5, 1.0, 0.0,  2.0, 2.0],
        [10.0, 12.0, 0.0, 0.2,  2.0, 2.0],
        [ 9.0, 12.5, 0.0, 0.5,  2.0, 2.0],
        [13.0,  3.0, 0.0, 0.0,  2.0, 2.0],
        [ 2.0,  1.0, 0.4, 0.0, 13.0,13.0],
        [ 5.0, 12.0, 0.0, 0.3, 13.0,13.0],
        [ 1.5,  8.0, 0.2, 0.0, 13.0,13.0],
        [ 2.2,  4.0, 0.0, 1.0, 13.0,13.0],
        [ 3.1, 13.0, 0.5, 0.0, 13.0,13.0],
    ])
    space = [
        np.array([(x, 0) for x in np.linspace(0, 15, 1000)]),
        np.array([(x,15) for x in np.linspace(0, 15, 1000)]),
        np.array([( 0, y) for y in np.linspace(0, 15, 1000)]),
        np.array([(15, y) for y in np.linspace(0, 15, 1000)])
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space), delta_t=0.1)
    states = np.stack([s.step().state.copy() for _ in range(1000)])

    print(space[0].shape)

    animate(states, space, "file")

if __name__ == "__main__":
    main()
