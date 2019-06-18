#!/usr/bin/env python
""" Evaluates a batch of blocks for stimuli generation.

Performs a series of analysis.

1. Physics differentials. (Optional)

Given an exhaustive dataframe (all configurations for a given set of towers),
determine how each configuration changes physical stability and direction of
falling.

2. Computes a histogram over the 2-D differential space.

Each configuration across towers is plotted where each dimension
is normalized by the average and variance of the entire pool.
"""

import os
import json
import glob
import string
import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from galileo_ramp.utils import config
from galileo_ramp.world.scene import ramp
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()


def get_collisions(ramp_file):
    """ Returns the time between the first two collisions

    Time in ms
    """
    with open(ramp_file, 'r') as f:
        data = json.load(f)
    state = forward_model.simulate(data['scene'], 900)
    state = state[-1]
    collided = np.sum(state, axis = -1) > 0
    indeces = np.flatnonzero(collided)
    print(state)
    print(indeces)
    delta = (indeces[1] - indeces[0])*(100./6.)
    return delta

def plot_profile(positions, results, out):
    """ Creates a 2D binary histogram of viable mass/positions.
    """
    fig, ax = plt.subplots(tight_layout=True,
                           figsize = (len(positions), 8))
    ax.violinplot(results, showmedians = True)
    ax.axhline(200)
    ax.set_xlabel('Position')
    ax.set_ylabel('Duration (ms)')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(positions) + 1))
    ax.set_xticklabels(positions)
    ax.set_xlim(0.25, len(positions) + 0.75)
    fig.savefig(out)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()

    out_path = CONFIG['PATHS', 'scenes']
    profile_path = os.path.join(out_path, 'collisions.png')
    print('Saving profile to {0!s}'.format(profile_path))
    pos_path = os.path.join(out_path, 'valid_positions.json')
    with open(pos_path, 'r') as f:
        positions = json.load(f)

    durations = []
    for pos in positions:
        pos_suffix = '1_*/{0:d}_*.json'.format(pos)
        scene_paths = glob.glob(os.path.join(out_path, pos_suffix))
        t = list(map(get_collisions, scene_paths))
        durations.append(t)

    plot_profile(positions, durations, profile_path)

if __name__ == '__main__':
   main()
