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


def plot_differences(full, mc):
    fig, axes = plt.subplots(nrows = 2)
    ax = axes[0]
    ax.set_title('Differences in position')
    # delta = full[0][:,0] - mc[0][:,0]
    # delta = np.linalg.norm(full[0] - mc[0], axis = -1)
    # ax.plot(delta)
    ax.plot(full[0][:,0])
    ax.plot(mc[0][:,0])
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
    t = 24
    full_trace = test_physics.run_full_trace(t, {"friction": 0.2})
    # mc_f = lambda s: test_physics.run_mc_trace(state = s)
    mc_trace = test_physics.run_mc_trace(T = t, pad = 0, data = {"friction": 0.01})
    print(full_trace[0][:, 0] - mc_trace[0][:, 0])
    plot_differences(full_trace, mc_trace)
    mc_trace = test_physics.run_mc_trace(T = t, pad = 0, data = {"friction": 0.2})
    print(full_trace[0][:, 0] - mc_trace[0][:, 0])


if __name__ == '__main__':
   main()
