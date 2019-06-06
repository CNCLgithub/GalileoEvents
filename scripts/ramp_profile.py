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
import sys
import glob
import copy
import json
import string
import hashlib
import argparse
import datetime
import numpy as np
import operator as op
from functools import reduce
from itertools import combinations, permutations


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from galileo_ramp.utils import config
from galileo_ramp.world.scene.ball import Ball
from galileo_ramp.world.scene.ramp import RampScene
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def profile_scene(appearances, radius, base_scene, n_ramp,
                   friction, pred, ramp_pcts, table_pcts, densities):
    """ Encapsulation for searching for interesting towers.

    Generates a random tower from the provided base and
    determine if it is an interesting tower.

    Returns:
    A tuple (passed, mut) where `passed` is `True` if the tower is interesting and
    `mut` is the mutation type (None if `passed == False`)
    """
    # Add balls to ramp scene
    scene = copy.deepcopy(base_scene)
    pcts = np.concatenate((np.array(ramp_pcts) + 1, table_pcts), axis = 0)
    for i in range(len(appearances)):
        ball = Ball(appearances[i], (radius,), densities[i], friction)
        scene.add_object(str(i), ball, pcts[i])

    # Eval predicate on trace
    trace = forward_model.simulate(scene.serialize(), 900)
    return pred(trace)


def predicate(state):
    """
    All unique collisions occur.

    WARNING: This is currently hard coded for the case
    of 3 balls, where the collision matrix as rows for:
    A->B, A->C, B->C
    """
    contacts = state[-1]
    return all(np.sum(contacts, axis = 0)[[0,-1]] > 0)

def read_csv_file(path):
    path = os.path.join(CONFIG['PATHS', 'scenes'], path)
    return np.genfromtxt(path, delimiter=',')

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def npr(n, r):
    """
    Calculate the number of ordered permutations of r items taken from a
    population of size n.

    >>> npr(3, 2)
    6
    >>> npr(100, 20)
    1303995018204712451095685346159820800000
    """
    assert 0 <= r <= n
    return np.product(np.arange(n - r + 1, n + 1))

def plot_profile(ratios, n_pos, results, out):
    """ Creates a 2D binary histogram of viable mass/positions.
    """
    fig, ax = plt.subplots(tight_layout=True)
    xs = np.repeat(np.arange(len(ratios)), n_pos)
    ys = np.tile(np.arange(n_pos), len(ratios))
    print(results[0])
    print(np.sum(results, axis = 2))
    ws = np.sum(results, axis = 2) == results.shape[2]
    hist = ax.hist2d(xs, ys, weights = ws.flatten(),
                     bins = (len(ratios), n_pos),
                     cmap = cm.binary)
    fig.savefig(out)
    plt.close(fig)

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('mass_file', type = str,
                        help = 'CSV file containing mass ratios')
    parser.add_argument('--table', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of table.')
    parser.add_argument('--table_steps', type = int, default = 4,
                        help = 'Number of positions along X-axis.')
    parser.add_argument('--ramp', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of ramp.')
    parser.add_argument('--ramp_steps', type = int, default = 4,
                        help = 'Number of positions along X-axis.')
    parser.add_argument('--ramp_angle', type = float, default = np.pi*(30.0/180),
                        help = 'ramp angle in degrees')
    parser.add_argument('--radius', type = float, default = 1.5,
                        help = 'Ball radius.')
    parser.add_argument('--friction', type = float, default = 0.4,
                        help = 'Ball friction')
    parser.add_argument('--n_ramp', type = int, default = 1,
                        help = 'Number of balls on ramp')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')
    parser.add_argument('--batch', type = int, default = 1,
                        help = 'Number of towers to search concurrently.')
    parser.add_argument('--debug', action = 'store_true',
                        help = 'Run in debug (no rejection).')
    args = parser.parse_args()

    name = os.path.basename(args.mass_file)[:-4]
    out_path = os.path.join(CONFIG['PATHS', 'scenes'], name)
    results_path = os.path.join(out_path, 'results.csv')
    profile_path = os.path.join(out_path, 'profile.png')
    print('Saving profile to {0!s}'.format(out_path))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # Digest masses
    mass_ratios = read_csv_file(args.mass_file)
    n_ratios, n_assignments = mass_ratios.shape[:]
    n_unique = len(np.unique(mass_ratios[0]))
    n_orderings = npr(n_assignments, n_unique)

    # Setup positions
    n_steps = args.ramp_steps + args.table_steps
    ramp_pcts = np.linspace(0.4, 0.9, args.ramp_steps)
    table_pcts = np.linspace(0.2, 0.7, args.table_steps)
    n_table = n_assignments - args.n_ramp
    n_positions = ncr(args.ramp_steps, args.n_ramp) * \
        ncr(args.table_steps, n_table)

    n_comb = n_orderings * n_positions

    base = RampScene(args.table, args.ramp,
                     ramp_angle = args.ramp_angle)

    params = {
        'appearances' : ['R', 'B', 'G'],
        'radius' : args.radius,
        'base_scene' : base,
        'n_ramp' : args.n_ramp,
        'friction' : args.friction,
        'pred' : predicate,
    }
    if os.path.isfile(results_path):
        results = np.load(results_path)
    else:
        results = np.zeros((n_ratios, n_positions, n_orderings))
        for row in range(n_ratios):
            ratio = mass_ratios[row]
            pi = 0
            for rp_pcts in combinations(ramp_pcts, args.n_ramp):
                for tb_pcts in combinations(table_pcts, n_table):
                    for mi, mass_assign in enumerate(permutations(ratio)):
                        d = {'ramp_pcts' : rp_pcts,
                             'table_pcts' : tb_pcts,
                             'densities' : mass_assign}
                        results[row, pi, mi] = profile_scene(**params, **d)
                    pi += 1

        np.save(results_path, results)
    plot_profile(mass_ratios, n_positions, results, profile_path)


if __name__ == '__main__':
   main()
