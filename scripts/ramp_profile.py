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
import copy
import json
import string
import hashlib
import argparse
import datetime
import numpy as np
import operator as op
from functools import reduce
from itertools import combinations
from sympy.utilities.iterables import multiset_permutations

from galileo_ramp.utils import config
from galileo_ramp.world.scene.ball import Ball
from galileo_ramp.world.scene.ramp import RampScene
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def profile_scene(appearances, radius, base_scene, n_ramp,
                   friction, pred, ramp_pcts, table_pcts, densities, out):
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
    result = {
        'scene' : scene.serialize(),
        # 'trace' : dict(zip(['pos', 'orn', 'lvl', 'avl', 'col'], trace))
    }
    r_str = json.dumps(result, indent = 4, sort_keys = True,
                       cls = forward_model.TraceEncoder)
    hashed = hashlib.md5()
    hashed.update(r_str.encode('utf-8'))
    hashed = hashed.hexdigest()
    out_file = out.format(hashed)
    print('Writing to ' + out_file)
    with open(out, 'w') as f:
        f.write(r_str)
    return pred(trace)


def predicate(state):
    """
    All unique collisions occur.

    WARNING: This is currently hard coded for the case
    of 3 balls, where the collision matrix as rows for:
    A->B, A->C, B->C
    """
    contacts = state[-1]
    print(np.sum(contacts, axis = 0))
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

def read_float(s):
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    else:
        return float(s)

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('ratio', type = read_float, nargs = '+',
                        help = 'Mass ratios (1 for each ball, may havel duplicates')
    parser.add_argument('--table', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of table.')
    parser.add_argument('--table_steps', type = int, default = 4,
                        help = 'Number of positions along X-axis.')
    parser.add_argument('--ramp', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of ramp.')
    parser.add_argument('--ramp_steps', type = int, default = 4,
                        help = 'Number of positions along X-axis.')
    parser.add_argument('--ramp_angle', type = float, default = 35,
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
    parser.add_argument('--fresh', action = 'store_true',
                        help = 'Ignore previous profiles')
    args = parser.parse_args()

    ratio_str = '-'.join(['{0:4.2f}'.format(m) for m in args.ratio])
    ratio_str = '{0:d}|{1!s}'.format(args.n_ramp, ratio_str)
    out_path = os.path.join(CONFIG['PATHS', 'scenes'], ratio_str)
    results_path = out_path + '.npy'
    print('Saving profile to {0!s}(.npy)'.format(out_path))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # Digest masses
    n_assignments = len(args.ratio)
    unique_ratio = np.unique(args.ratio)
    n_unique = len(unique_ratio)
    n_orderings = npr(n_assignments, n_unique - 1)

    # Setup positions
    n_steps = args.ramp_steps + args.table_steps
    ramp_pcts = np.linspace(0.9, 0.4, args.ramp_steps)
    table_pcts = np.linspace(0.2, 0.7, args.table_steps)
    n_table = n_assignments - args.n_ramp
    n_positions = ncr(args.ramp_steps, args.n_ramp) * \
        ncr(args.table_steps, n_table)

    n_comb = n_orderings * n_positions

    base = RampScene(args.table, args.ramp,
                     ramp_angle = args.ramp_angle * (np.pi/180.))

    params = {
        'appearances' : ['R', 'B', 'G'],
        'radius' : args.radius,
        'base_scene' : base,
        'n_ramp' : args.n_ramp,
        'friction' : args.friction,
        'pred' : predicate,
    }
    results = np.zeros((n_positions, n_orderings))
    pi = 0
    if os.path.isfile(results_path) and not args.fresh:
        print('Already computed, exiting')
        return

    for rp_pcts in combinations(ramp_pcts, args.n_ramp):
        for tb_pcts in combinations(table_pcts, n_table):
            for mi, mass_assign in enumerate(
                    multiset_permutations(args.ratio, n_assignments)
            ):
                out_scene = '{0:d}_{1:d}.json'.format(pi,mi)
                out_scene = os.path.join(out_path, out_scene)
                d = {'ramp_pcts' : rp_pcts,
                     'table_pcts' : tb_pcts,
                     'densities' : mass_assign,
                     'out' : out_scene}
                results[pi, mi] = profile_scene(**params, **d)

            pi += 1

    np.save(results_path, results)


if __name__ == '__main__':
   main()
