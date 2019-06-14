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
from pprint import pprint
from functools import reduce
from itertools import combinations, permutations

from galileo_ramp.utils import config
from galileo_ramp.world.scene.ball import Ball
from galileo_ramp.world.scene.ramp import RampScene
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def profile_scene(appearances, radius, base_scene, n_ramp,
                   friction, ramp_pcts, table_pcts, densities):
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


    s = forward_model.simulate(scene.serialize(), 900, debug =True)
    print(np.sum(s[-1], axis = 0))


def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates the energy of mass ratios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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


    # Setup positions
    ramp_pcts = np.array([0.9])
    table_pcts = np.array([0.2, 0.35,])
    mass_assign = np.array((1, 2, 2))

    base = RampScene(args.table, args.ramp,
                     ramp_angle = args.ramp_angle * (np.pi/180.))

    params = {
        'appearances' : ['R', 'B', 'G'],
        'radius' : args.radius,
        'base_scene' : base,
        'n_ramp' : args.n_ramp,
        'friction' : args.friction,
    }
    d = {'ramp_pcts' : ramp_pcts,
         'table_pcts' : table_pcts,
         'densities' : mass_assign}
    profile_scene(**params, **d)



if __name__ == '__main__':
   main()
