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
from blockworld.utils.json_encoders import TowerEncoder

from utils import config
from experiment.generator.noisy_gen import PushedSim
from experiment.generator.multi_gen import MultiBlockGen

CONFIG = config.Config()

def evaluate_tower(base, size, gen, phys, pred, out,
                   materials = None, force = 1.0, debug = False, tries = 10):
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
    # Build base tower
    tower = gen(base, size)
    # add mutations in density
    mutations = list(gen.configurations(tower, materials))[:tries]
    # compute stats over towers
    orig_trace, orig_stats, _ = phys.analyze(tower, force = force)
    # Don't continue if query[0] fails
    # (no combinations will satisfy constraint)
    pprint(orig_stats)
    if not pred['orig'](orig_stats):
        return (False, None)
    print('Original')
    passed = False
    hashed = None
    for (block_ids, assignments) in mutations:
        for mut in assignments:
            mut_tower = assignments[mut]
            mut_trace, mut_stats, _ = phys.analyze(mut_tower, force = force)
            pprint(mut_stats)
            passed = debug or pred['mut'](mut_stats)
            print(passed)
            if passed:
                metric = pred['mode']
                value = mut_stats[metric] - orig_stats[metric]
                # Package results for json
                result = {
                    'org-tower' : {
                        'struct': tower.serialize(),
                        'stats' : orig_stats,
                        'trace' : orig_trace,
                    },
                    'mut-tower' : {
                        'struct' : mut_tower.serialize(),
                        'stats'  : mut_stats,
                        'trace' : mut_trace,
                    },
                    'block' : block_ids.tolist(),
                    'mutation' : mut,
                    'metric' : metric,
                    'metric_delta' : value
                }
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

    return (passed, hashed)

def make_stab_pred(args):
    mode = 'instability'
    metric = 'instability_diff'
    d = {
        'orig' : (lambda s: (s['instability'] == 0.0) and \
                  (s['instability_mu'] < args.upper)),
        'mut'  : (lambda s: (s['instability'] > args.lower) and \
                  (s['instability_mu'] >= args.noisy)),
        'mode' : 'instability',
        'metric' : 'instability_diff',
    }
    return d

def make_unst_pred(args):
    mode = 'instability'
    metric = 'instability_diff'
    d = {
        'orig'  : (lambda s: (s['instability'] > args.lower) and \
                  (s['instability_mu'] >= args.noisy)),
        'mut' : (lambda s: (s['instability'] == 0.0) and \
                  (s['instability_mu'] < args.upper)),
        'mode' : 'instability',
        'metric' : 'instability_diff',
    }
    return d

def stab_path(args):
    return 'stable_{}_{}_{}_{}'.format(args.weight, args.upper, args.lower,
                                       args.noisy)
def unst_path(args):
    return 'unstable_{}_{}_{}_{}'.format(args.weight, args.upper, args.lower,
                                       args.noisy)

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--total', type = int, help = 'Number of towers to generate',
                        default = 1)
    parser.add_argument('--size', type = int, help = 'The size of each tower.',
                        default = 10)
    parser.add_argument('--base', type = int, nargs = 2, default = (2,2),
                        help = 'Dimensions of base.')
    parser.add_argument('--shape', type = int, nargs = 3, default = (3,1,1),
                        help = 'Dimensions of block (x,y,z).')
    parser.add_argument('--out', type = str, help = 'Path to save towers.')
    parser.add_argument('--base_path', type = str,
                        help = 'Path to base tower.')
    parser.add_argument('--noise', type = float, default = 0.2,
                        help = 'Noise to add to positions.')
    parser.add_argument('--force', type = float, default = 250.0,
                        help = 'Force to push blocks')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')
    parser.add_argument('--batch', type = int, default = 1,
                        help = 'Number of towers to search concurrently.')
    parser.add_argument('--n_blocks', type = int, default = 2,
                        help = 'The number of blocks to mutate at a time.')
    parser.add_argument('--debug', action = 'store_true',
                        help = 'Run in debug (no rejection).')
    parser.add_argument('weight', type=str, choices = ['L', 'H'],
                             help = 'Which mutation to search over')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_stab = subparsers.add_parser('stable',
                                        help='Search for pairs where the original is stable')
    parser_stab.add_argument('--upper', type=float, default = 0.4,
                             help = ('Upper bound for original instability'))
    parser_stab.add_argument('--lower', type=float, default = 0.4,
                             help = ('Minimum instability for mutations'))
    parser_stab.add_argument('--noisy', type=float, default = 0.4,
                             help = ('Minimum average instability for mutations'))
    parser_stab.set_defaults(search=make_stab_pred, make_path = stab_path)

    parser_unst = subparsers.add_parser('unstable',
                                        help='Search for pairs where the original is unstable')
    parser_unst.add_argument('--upper', type=float, default = 0.4,
                             help = ('Maximum instability for mutations'))
    parser_unst.add_argument('--lower', type=float, default = 0.4,
                             help = ('Lower bound for original instability'))
    parser_unst.add_argument('--noisy', type=float, default = 0.4,
                             help = ('Minimum average instability for original'))
    parser_unst.set_defaults(search=make_unst_pred, make_path = unst_path)
    args = parser.parse_args()

    if args.out is None:
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out_d = args.make_path(args) + '_' + suffix
        out_d = os.path.join(CONFIG['PATHS', 'towers'], out_d)
    else:
        out_d = os.path.join(CONFIG['PATHS', 'towers'], args.out)

    print('Saving new towers to {0!s}'.format(out_d))

    if args.base_path is None:
        base = args.base
        base_path = '{0:d}x{1:d}'.format(*base)
    else:
        base = towers.simple_tower.load(args.base_path)
        out_d += '_extended'
        base_path = os.path.basename(os.path.splitext(args.base_path)[0])

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
