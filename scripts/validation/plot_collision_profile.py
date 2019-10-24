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
    print(ramp_file)
    print(indeces)
    delta = (indeces[1] - indeces[0])*(100./6.)
    return delta

def plot_profile(positions, results, out):
    """ Creates a 2D binary histogram of viable mass/positions.
    """
    fig, axes = plt.subplots(nrows = 2, tight_layout=True,
                           figsize = (len(positions), 16))
    ax = axes[0]
    r = ax.violinplot(results, showmedians = True)
    ax.axhline(200)
    ax.set_xlabel('Position')
    ax.set_ylabel('Duration (ms)')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(positions) + 1))
    ax.set_xticklabels(positions)
    ax.set_xlim(0.25, len(positions) + 0.75)
    ax = axes[1]
    ax.scatter(positions, np.min(results, axis = 1))
    ax.axhline(300)
    ax.set_xlabel('Position')
    ax.set_ylabel('Minimum Duration (ms)')
    fig.savefig(out)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('on_ramp', type = int,
                        help = 'Number of balls on ramp')
    args = parser.parse_args()

    out_path = CONFIG['PATHS', 'scenes']
    profile_path = '{0:d}_collisions.png'.format(args.on_ramp)
    profile_path = os.path.join(out_path, profile_path)

    print('Saving profile to {0!s}'.format(profile_path))

    pos_path = '{0:d}_valid_positions.json'.format(args.on_ramp)
    pos_path = os.path.join(out_path, pos_path)
    with open(pos_path, 'r') as f:
        positions = json.load(f)

    durations = []
    pos_suffix = '{0!s}_*/'.format(args.on_ramp) + '{0:d}_*.json'
    for pos in positions:
        scene_paths = glob.glob(os.path.join(out_path, pos_suffix.format(pos)))
        t = list(map(get_collisions, scene_paths))
        durations.append(t)

    plot_profile(positions, durations, profile_path)

    dur_path = '{0:d}_collision_durations.json'.format(args.on_ramp)
    dur_path = os.path.join(out_path, dur_path)
    with open(dur_path, 'w') as f:
        json.dump(dict(zip(positions, durations)), f)

if __name__ == '__main__':
   main()
