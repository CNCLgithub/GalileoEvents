#!/usr/bin/env python3
""" Evaluates the convergence statistics across several free parameter settings

Given a collection of free parameters, determines the probability
that the generated chains result in a ground truth mass ratio judgment.

"""
import os
import json
import glob
import h5py
import argparse
import numpy as np
from scipy import stats

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

from experiment.inference import trace
from experiment.dataset.particle_dataset import ParticleDataset

from utils import config
CONFIG = config.Config()

def retrieve_parameter(run, param):
    path = os.path.join(run, 'parameters.json')
    with open(path, 'r') as f:
        params = json.load(f)
    return params[param]

def bootstrap_timings(src, ref, conf = 0.96, size = 1000):
    n_frames = src.shape[0]
    for t in np.arange(n_frames):
        x = np.random.choice(src[t], size = size)
        y = np.random.choice(ref[t], size = size)
        q = np.mean((x - y) > 0)
        print((t, q))
        if q > conf:
            return t
    return n_frames

def plot_pf_ll(trace_path, ref_path, out):

    estimates, scores, result = trace.extract_chains(trace_path, maps = False)
    _, ref, _ = trace.extract_chains(ref_path, maps = False)
    xs = result['xs']

    fig, axes = plt.subplots(2, 1, figsize = (8,8))

    # Plot trace timings
    ax = axes[0]
    ax.set_xlabel('Time')
    ax.set_ylabel('Scores')
    vparts = ax.violinplot(scores.T, xs, widths = 6.0, showmedians = True)
    for pc in vparts['bodies']:
        pc.set_color('cyan')

    vparts = ax.violinplot(ref.T, xs, widths = 6.0, showmedians = True)
    for pc in vparts['bodies']:
        pc.set_color('orange')

    t0 = bootstrap_timings(scores, ref)
    indeces = np.geomspace(max(0.01, t0), len(xs)-1, num = 4, endpoint = True)
    indeces = indeces.astype(int).tolist()
    times = xs[indeces]
    print(t0)
    print(times)
    colors = cm.rainbow(np.linspace(0,1, len(times) + 2))
    for tdx, time in enumerate(times):
        ax.axvline(time, label = 't-{}'.format(tdx), c = colors[tdx])

    fig.legend()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    # plot model predictions
    ax = axes[1]
    ax.set_xlabel('Time')
    ax.set_ylabel('Log-Mass Ratio')

    mass = np.log(estimates[0]) - np.log(3)
    if result['gt'][0] > 3.0:
        ax.set_ylim(np.log([1, 13]) - np.log(3))
    # else:
    #     ax.set_ylim([-5, 0.1])

    # ax.errorbar(np.arange(len(times)), np.mean(mass[indeces, :], axis = 1),
    #        yerr = np.std(mass[indeces, :], axis = 1))
    ax.plot(np.arange(len(times)), np.mean(mass[indeces, :], axis = 1))

    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return times.tolist()

def mean_confidence_interval(data, confidence=0.95):
    """
    cribbed from:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def main():

    parser = argparse.ArgumentParser(
        description = 'Computes the ')
    parser.add_argument('run', type = str,
                        help = 'Paths to the PF runs')
    parser.add_argument('reference', type = str,
                        help = 'Paths to the reference runs')
    parser.add_argument('--trials', type = int, nargs = '+',
                        help = 'Specific trials to run.')
    args = parser.parse_args()

    run = os.path.join(CONFIG['PATHS','traces'], args.run)
    reference = os.path.join(CONFIG['PATHS','traces'], args.reference)
    dataset_path = os.path.join(CONFIG['PATHS', 'databases'],
                                retrieve_parameter(run, 'dataset'))
    dataset = ParticleDataset(dataset_path)
    if args.trials is None:
        trials = np.arange(len(dataset))
    else:
        trials = args.trials

    time_points = {}
    for trial in trials:
        src = os.path.join(run, 'trial_{0:d}.hdf5'.format(trial))
        ref = os.path.join(reference, 'trial_{0:d}.hdf5'.format(trial))
        if (not os.path.isfile(src)) or (not os.path.isfile(ref)):
            print('File {} not found, skipping trial {}'.format(src, trial))
            continue
        out_path = os.path.join(run, 'trial_{0:d}_timing.png'.format(trial))
        t0 = plot_pf_ll(src, ref, out_path)
        time_points[str(trial)] = t0

    out = os.path.join(run, 'time_points.json')
    with open(out, 'w') as f:
        json.dump(time_points, f)

if __name__ == '__main__':
    main()
