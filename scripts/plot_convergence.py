#!/usr/bin/env python3
"""
Visualizes inference estimates (MAPs)
"""

import os
import json
import argparse
import numpy as np

from galileo_ramp.utils import config
from galileo_ramp.inference import trace
CONFIG = config.Config()

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm



def plot_convergence(trace_path, out, collapse = False):
    """ Computes statistics over inference trace


    Arguments:
        trace (io.BytesIO or str): Reference to inference trace.

    Returns
        metrics (dict): A dictionary containing several metrics such
        as time to convergence.
    """
    estimates, scores, result = trace.extract_chains(trace_path, maps = collapse)
    print(estimates)
    estimates = np.exp(estimates)
    print(estimates)
    xs = result['xs']
    latents = result['latents']
    fig, axes = plt.subplots(nrows=estimates.shape[0] + 1,
                             figsize=(8, 8))
    print(xs)
    for row, latent in enumerate(latents):
        latent_str = ' => '.join(map(lambda x: x.decode('UTF-8'),
                                     latent))
        ax = axes[row]
        ax.violinplot(estimates[row].T, xs,
                      widths = 15.0, showmedians = True)
        ax.set_ylabel(latent_str)
        ax.set_xlabel('Time')
        ax.set_ylim((0, 6))
        gt = result['gt'][row]
        ax.axhline(gt, lw=1.7, ls='--', label='GT')

    ax = axes[row + 1]
    ax.violinplot(scores.T, xs, widths = 6.0, showmedians = True)
    ax.set_title('Scores')
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
    parser.add_argument('test', type = str, help = 'Test name')
    parser.add_argument('run', type = str, help = 'Path to the PF run')
    parser.add_argument('--collapse', action = 'store_true',
                        help = 'Collapse particles across chains')
    args = parser.parse_args()

    out_dir = os.path.join(CONFIG['PATHS','traces'], args.test)
    run = os.path.join(out_dir, args.run)
    if not os.path.isfile(run):
        print('File {} not found'.format(run))
        return
    out_path = run.replace('.hdf5', '.png')
    print(out_path)
    plot_convergence(run, out_path, collapse = args.collapse)

if __name__ == '__main__':
    main()
