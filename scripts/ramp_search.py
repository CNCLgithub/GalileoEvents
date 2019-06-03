#!/bin/python3
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
import hashlib
import argparse
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from dask import distributed
from dask_jobqueue import SLURMCluster

from utils import config

CONFIG = config.Config()

def evaluate_tower(materials, densities, base_scene, debug = False):
    """ Encapsulation for searching for interesting towers.

    Generates a random tower from the provided base and
    determine if it is an interesting tower.

    Arguments:
    - base (dict): A `dict` containing the fields:
                 {
                   'base' : Tower base to build over (can be `None`)
                   'k'    : The number of blocks to add
                   'idx'  : ...
                 }
    - gen (blockworld.Generator): Creates towers and their configurations
    - phys (blockworld.TowerEntropy): Evaluates the physical properties of towers
    - mode (str): Use 'angle' for difference in angle and 'instability' for instability

    Returns:
    A tuple (passed, mut) where `passed` is `True` if the tower is interesting and
    `mut` is the mutation type (None if `passed == False`)
    """
    # Assign materials
    mat_ids = np.random.permute(len(materials))
    dens_ids = np.random.permute(len(densities))

    # balls on ramp
    rng = np.linalg(0.1, 0.9, 4)
    ramp_pcts = np.random.choice(rng, n_ramp, replace = False)
    table_pcts = np.random.choice(rng, n - n_ramp, replace = False)
    pcts = np.concatenate((table_pcts, ramp_pcts + 1))
    scene = copy(base_scene)
    for i, (m_id, d_id) in enumerate(zip(mat_ids, dens_ids)):
        ball = Ball(materials[m_id], (radius,), densities[d_id], friction)
        scene.add_object(names[i], ball, pcts[i])

    stats = simulate(scene)
    if not pred(stats):
        return (False, None)
    # Save the resulting tower pair using the content of `result` for
    # the hash / file name
    r_str = json.dumps(result, indent = 4, sort_keys = True,
                            cls = TowerEncoder)
    hashed = hashlib.md5()
    hashed.update(r_str.encode('utf-8'))
    hashed = hashed.hexdigest()
    out_file = out.format(hashed)
    print('Writing to ' + out_file)
    with open(out_file, 'w') as f:
        f.write(r_str)
    return (passed, hashed)


def predicate(state):
    """
    Last object moves
    """
    return np.sum(state['lin_vel'][:, -1]) > 0

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates a random ramp scenes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--total', type = int, help = 'Number of scenes to generate',
                        default = 1)
    parser.add_argument('--size', type = int, help = 'The number of balls.',
                        default = 3)
    parser.add_argument('--table', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of table.')
    parser.add_argument('--ramp', type = int, nargs = 2, default = (35, 18),
                        help = 'XY dimensions of ramp.')
    parser.add_argument('--shape', type = int, nargs = 3, default = (3,1,1),
                        help = 'Dimensions of block (x,y,z).')
    parser.add_argument('--out', type = str, help = 'Path to save towers.')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')
    parser.add_argument('--batch', type = int, default = 1,
                        help = 'Number of towers to search concurrently.')
    parser.add_argument('--debug', action = 'store_true',
                        help = 'Run in debug (no rejection).')
    args = parser.parse_args()

    if args.out is None:
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out_d = args.make_path(args) + '_' + suffix
        out_d = os.path.join(CONFIG['PATHS', 'scenes'], out_d)
    else:
        out_d = os.path.join(CONFIG['PATHS', 'scenes'], args.out)

    print('Saving new scenes to {0!s}'.format(out_d))

    if os.path.isdir(out_d):
        t = glob.glob(os.path.join(out_d, '*.json'))
        t = list(map(lambda x: x[:-5], t))
        results = np.full(args.total, '', dtype = object)
        results[:len(t)] = t
    else:
        os.mkdir(out_d)
        results = np.full(args.total, '', dtype = object)

    out_path = os.path.join(out_d, '{0!s}.json')
    enough_scenes = lambda l: len(np.argwhere(l == '')) == 0
    predicate = args.search(args)

    materials = {'Wood' : 1.0}
    gen = MultiBlockGen(materials, 'local', args.shape, args.n_blocks)
    phys = PushedSim(noise = args.noise, frames = 240)

    params = {
        'base' : base,
        'size' : args.size,
        'gen'  : gen,
        'phys' : phys,
        'pred' : predicate,
        'out'  : out_path,
        'materials' : [args.weight],
        'force' : args.force,
        'debug': args.debug,
    }
    eval_tower = lambda x: evaluate_tower(**params)

    if enough_scenes(results):
        print('All done')
    else:
        client = initialize_dask(args.batch, slurm = args.slurm)
        # Submit first batch of towers
        ac = distributed.as_completed(
            client.map(eval_tower,
                       np.repeat(None, args.batch),
                       pure = False)
        )
        for future in ac:
            # Retrieve future's result
            (passed, hashed_path) = future.result()
            # Determine if the tower had interesting configurations
            if passed:
                check = np.argwhere(results == '')
                if len(check) == 0:
                    print('Removing', hashed_path)
                    os.remove(out_path.format(hashed_path))
                else:
                    results[check[0]] = hashed_path
                    print('Added {}'.format(hashed_path))

            # Delete future for good measure
            client.cancel(future)
            # Check to see if more trials are needed
            n_pending = ac.count()
            if not enough_scenes(results) and (n_pending < args.batch):
                ac.update(
                    client.map(eval_tower,
                               np.repeat(None, args.batch - n_pending),
                               pure = False)
                )
            else:
                break

    # Close the client
    client.close()


def initialize_dask(n, factor = 5, slurm = False):

    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)

    else:
        n = min(500, n)
        py = './run.sh python3'
        params = {
            'python' : py,
            'cores' : 1,
            'memory' : '1GB',
            'walltime' : '1-0',
            'processes' : 1,
            'job_extra' : [
                '--qos use-everything',
                # '--qos tenenbaum',
                '--array 0-{0:d}'.format(n - 1),
                '--requeue',
                '--output "/dev/null"'
                # ('--output ' + os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out')),
            ],
            'env_extra' : [
                'JOB_ID=${SLURM_ARRAY_JOB_ID%;*}_${SLURM_ARRAY_TASK_ID%;*}',
                'source /etc/profile.d/modules.sh',
                'cd {0!s}'.format(CONFIG['PATHS', 'root']),
            ]
        }
        cluster = SLURMCluster(**params)
        print(cluster.job_script())
        cluster.scale(1)

    print(cluster.dashboard_link)
    return distributed.Client(cluster)


if __name__ == '__main__':
   main()
