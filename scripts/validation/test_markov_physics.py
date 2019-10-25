#!/usr/bin/env python3

"""
Evaluates the difference between running a simulation
from 0->t in a markovian format
"""
import os
import glob
import json
import argparse
import datetime
import numpy as np
from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from galileo_ramp.utils import config
from galileo_ramp.world.simulation import test_physics

CONFIG = config.Config()

T = 24
def run_full_trace():
    t = test_physics.TestPhysics()
    return t.get_trace(T, fps = 6)

def run_mc_trace(debug, pad = 1, fps = 6):
    t = test_physics.TestPhysics(debug = debug)
    traces = []
    steps = 2*pad + 1
    current_trace = t.get_trace(steps, fps = fps)
    smooth_trace = lambda x: x[0]
    traces.append(list(map(smooth_trace, current_trace)))
    for _ in range(T-1):
        current_trace = t.get_trace(steps, state = current_trace, fps = fps)
        traces.append(list(map(smooth_trace, current_trace)))

    return list(map(np.vstack, zip(*traces)))

def plot_differences(full, mc):
    fig, axes = plt.subplots(nrows = 2)
    ax = axes[0]
    ax.set_title('Differences in position')
    # delta = full[0][:,0] - mc[0][:,0]
    delta = np.linalg.norm(full[0] - mc[0], axis = -1)
    ax.plot(delta)
    # ax.plot(full[0][:,0])
    # ax.plot(mc[0][:,0])
    ax = axes[1]
    ax.set_title('Differences in velocity')
    ax.set_xlabel('Time')
    # delta =full[2][:, 0] - mc[2][:, 0]
    delta = np.linalg.norm(full[2] - mc[2], axis = -1)
    ax.plot(delta)
    fig.savefig('test.png')
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description = 'Runs test physics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--debug', action = 'store_true',
                        help = 'Run in debug')

    args = parser.parse_args()

    full_trace = run_full_trace()
    mc_trace = run_mc_trace(args.debug)
    plot_differences(full_trace, mc_trace)


if __name__ == '__main__':
   main()
