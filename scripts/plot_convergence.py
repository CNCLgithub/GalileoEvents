#!/usr/bin/env python3
"""
Visualizes inference estimates (MAPs)
"""

import os
import json
import argparse
import numpy as np

from utils import config
CONFIG = config.Config()
from experiment.inference import trace
from experiment.dataset.particle_dataset import ParticleDataset

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm



def plot_convergence(trace_path, blocks, out, collapse = False):
    """ Computes statistics over inference trace


    Arguments:
        trace (io.BytesIO or str): Reference to inference trace.

    Returns
        metrics (dict): A dictionary containing several metrics such
        as time to convergence.
    """
    estimates, scores, result = trace.extract_chains(trace_path, maps = collapse)
    xs = result['xs']
    fig, axes = plt.subplots(nrows=estimates.shape[0] + 1,
                             figsize=(8, 8))
    ax = axes[0]
    ax.violinplot(estimates[0].T, xs, widths = 6.0, showmedians = True)
    block_str = ', '.join(map(str, blocks))
    ax.set_title('Estimates for Blocks {}'.format(block_str))
    ax.set_ylabel('Mass')
    ax.set_xlabel('Time')
    ax.set_ylim((0, 13))
    ax.axhline(result['gt'][0], lw=1.7, ls='--', label='GT')

    ax = axes[1]
    ax.violinplot(scores.T, xs, widths = 6.0, showmedians = True)
    block_str = ', '.join(map(str, blocks))
    ax.set_title('Scores for Blocks {}'.format(block_str))
    ax.set_ylabel('Log Score')
    ax.set_xlabel('Time')

    fig.legend()
    fig.savefig(out)
    plt.close(fig)


def retrieve_parameter(run, param):
    path = os.path.join(run, 'parameters.json')
    with open(path, 'r') as f:
        params = json.load(f)
    return params[param]

def main():

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = 'Visualizes inference traces')
    parser.add_argument('run', type = str,
                        help = 'Paths to the PF runs')
    parser.add_argument('--trials', type = int, nargs = '+',
                        help = 'Specific trials to run.')
    parser.add_argument('--collapse', action = 'store_true',
                        help = 'Collapse particles across chains')
    args = parser.parse_args()

    run = os.path.join(CONFIG['PATHS','traces'], args.run)
    dataset_path = os.path.join(CONFIG['PATHS', 'databases'],
                                retrieve_parameter(run, 'dataset'))
    dataset = ParticleDataset(dataset_path)
    if args.trials is None:
        trials = np.arange(len(dataset))
    else:
        trials = args.trials

    for trial in trials:
        file = os.path.join(run, 'trial_{0:d}.hdf5'.format(trial))
        if not os.path.isfile(file):
            print('File {} not found, skipping trial {}'.format(file, trial))
            continue
        out_path = os.path.join(run, 'trial_{0:d}_estimates.png'.format(trial))
        print(out_path)
        (_, blocks), _ = dataset[trial]
        plot_convergence(file, blocks, out_path, collapse = args.collapse)

if __name__ == '__main__':
    main()
