#!/usr/bin/env python

""" Loads a scene json directly into pybullet
"""

import os
import json
import argparse
import numpy as np
from pyquaternion import Quaternion
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from galileo_ramp.utils import config
from galileo_ramp.world.simulation import exp2_physics

CONFIG = config.Config()

def simulate_scene(src):
    """ Encapsulation for searching for interesting towers.

    Generates a random tower from the provided base and
    determine if it is an interesting tower.

    Returns:
    A tuple (passed, mut) where `passed` is `True` if the tower is interesting and
    `mut` is the mutation type (None if `passed == False`)
    """
    with open(src, 'r') as f:
        scene_data = json.load(f)['scene']
    s = exp2_physics.run_full_trace(scene_data,
                                    ["A", "B"],
                                    0.95,
                                    fps = 60,
                                    time_scale = 10.0)
    return s
    # print(np.sum(s[-1], axis = 0))

def compare_simulations(legacy, current):
    # align matrix dimensions
    shape = legacy.shape
    dur = shape[1]
    current = current[:dur].swapaxes(0, 1)
    # rotate the coordinates by 180
    # quat = Quaternion(axis=[0, 0, 1], angle=np.pi)
    # for i in range(dur):
    #     current[0, i, :] = quat.rotate(current[0, i, :])
    #     current[1, i, :] = quat.rotate(current[1, i, :])
    # align coordinate space
    print(legacy[:, 0, :])
    print(current[:, 0, :])
    ref = current[:, 0, :] - legacy[:, 0, :]
    print(ref)

    dl = np.linalg.norm(legacy[1] - legacy[0], axis = -1)
    cl = np.argmin(dl)
    dc = np.linalg.norm(current[1] - current[0], axis = -1)
    cc = np.argmin(dc)

    fig, ax = plt.subplots()
    ax.set_title('Difference in Distance between B -> A')
    ax.set_ylabel('L2')
    ax.set_xlabel('Frame')
    ax.plot(dl - dc, label = 'legacy - dc')
    ax.axvline(cl, label = 'collision-legacy')
    ax.axvline(cc, label = 'collision-current')
    fig.legend()
    fig.savefig('trace-distances.png')
    plt.close(fig)

    print(cl)
    print(cc)

    fig, axes = plt.subplots(nrows = 2)
    ax = axes[0]
    ax.set_title('Unaligned Phyiscs Traces')
    ax.set_ylabel('X')
    ax.set_xlabel('Frame')
    for i in range(2):
        ax.plot(legacy[i, :, 0], label = 'legacy_{0:d}'.format(i))
        ax.plot(current[i, :, 0], label = 'current_{0:d}'.format(i))

    ax = axes[1]
    ax.set_ylabel('Z')
    ax.set_xlabel('Frame')
    for i in range(2):
        ax.plot(legacy[i, :, 2])
        ax.plot(current[i, :, 2])
    fig.legend()
    fig.savefig('trace-diff-unaligned.png')
    plt.close(fig)

    legacy[0] = legacy[0, :] + ref[0]
    legacy[1] = legacy[1, :] + ref[1]
    fig, axes = plt.subplots(nrows = 2)
    ax = axes[0]
    ax.set_title('Aligned Phyiscs Traces')
    ax.set_ylabel('X')
    ax.set_xlabel('Frame')
    for i in range(2):
        ax.plot(legacy[i, :, 0], label = 'legacy_{0:d}'.format(i))
        ax.plot(current[i, :, 0], label = 'current_{0:d}'.format(i))

    ax = axes[1]
    ax.set_ylabel('Z')
    ax.set_xlabel('Frame')
    for i in range(2):
        ax.plot(legacy[i, :, 2])
        ax.plot(current[i, :, 2])
    fig.legend()
    fig.savefig('trace-diff-aligned.png')
    plt.close(fig)

    # print(distances.shape)
    # print(distances)
    # print(np.mean(distances, axis = -1))

def main():

    parser = argparse.ArgumentParser(
        description = 'Loads a scene json directly into pybullet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str,
                        help = 'Path to json')

    args = parser.parse_args()

    position_file = args.src.replace('.json', '_pos.npy')
    trace = np.load(position_file)
    sim = simulate_scene(args.src)
    compare_simulations(trace, sim[0])


if __name__ == '__main__':
   main()
