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

CONFIG = config.Config()


def plot_profile(ratios, results, out):
    """ Creates a 2D binary histogram of viable mass/positions.
    """
    n_rat, n_pos = results.shape[:2]
    fig, ax = plt.subplots(tight_layout=True, figsize = (n_pos, n_rat))
    xs = np.tile(np.arange(n_pos), n_rat)
    ys = np.repeat(np.arange(n_rat), n_pos)
    hist = ax.hist2d(xs, ys, weights = results.flatten(),
                     bins = (n_pos + 1, n_rat),
                     cmap = cm.binary)
    ax.xaxis.set_ticks(np.arange(n_pos + 1))
    ax.yaxis.set_ticks(np.arange(n_rat))
    tickNames = plt.setp(ax, yticklabels=ratios)
    plt.setp(tickNames, rotation=45, fontsize=8)
    fig.savefig(out)
    plt.close(fig)

def process_result(array):
    return np.sum(array, axis = -1) == array.shape[-1]

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()

    out_path = CONFIG['PATHS', 'scenes']
    profile_path = os.path.join(out_path, 'profile.png')
    result_paths = glob.glob(os.path.join(out_path, '*.npy'))
    result_paths = sorted(result_paths)
    names = list(map(lambda x: os.path.basename(x[:-4]),
                     result_paths))
    print('Saving profile to {0!s}'.format(profile_path))

    results = map(np.load, result_paths)
    results = map(process_result, results)
    results = np.array(list(results))

    plot_profile(names, results, profile_path)

    positions = np.argwhere(np.sum(results, axis = 0) == results.shape[0])
    with open(os.path.join(out_path, 'valid_positions.json'), 'w') as f:
        json.dump(positions.flatten().tolist(), f)

if __name__ == '__main__':
   main()
