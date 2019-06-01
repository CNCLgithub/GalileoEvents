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
# from stimuli.generate_towers import MyGenerator
from blockworld.utils.json_encoders import TowerEncoder
from stimuli.analyze_direction import ExpGen, ExpStability

CONFIG = config.Config()

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)

def idxquantile(s, q=0.5, *args, **kwargs):
    """ Returns the specified quantile.
    """
    qv = s.quantile(q, *args, **kwargs)
    return (s.sort_values()[::-1] <= qv).idxmax()


def cos_2vec(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mutate_tower(t, block):
    """ Changes the given block in the tower to an unknown appearance
    """
    aps = t.extract_feature('appearance')
    aps[int(block)] = 'U'
    t = t.apply_feature('appearance', aps)
    return t

def add_cols(row):
    parts = row['id'].split('_')
    row['block'] = int(parts[0])
    row['mut'] = parts[1]
    return row


def evaluate_tower(base, size, gen, phys, pred, out, debug = False):
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
    # compute stats over towers
    stats = phys(tower)[0]
    # evaluate metrics
    columns = ('id', 'angle', 'mag', 'instability', 'instability_p')
    print(stats['instability'], np.mean(stats['instability_p']))
    if not pred(stats):
        return (False, None)

    result = {
        'struct': tower.serialize(),
        'stats' : {k: stats[k] for k in columns},
        'trace' : stats['trace'],
    }
    # Package results for json
    # Save the resulting tower using the content of `result` for
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
        return (True, hashed)

def create_predicate(mode, thresh):
    if mode == 'stable':
        pred = lambda t: all([
            t['instability'] == 0,
            (np.mean(t['instability_p']) < thresh)])
    else:
        pred = lambda t: all([
            t['instability'] >= thresh,
            (np.mean(t['instability_p']) > thresh)])
    return pred

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
    parser.add_argument('--noise', type = float, default = 0.04,
                        help = 'Noise to add to positions.')
    parser.add_argument('--metric', type = str, default = 'stable',
                        choices = ['stable', 'unstable'],
                        help = 'Type of predicate.')
    parser.add_argument('--threshold', type = float, default = 0.0,
                        help = 'Upper bound of instability.')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')
    parser.add_argument('--batch', type = int, default = 1,
                        help = 'Number of towers to search concurrently.')
    args = parser.parse_args()

    if args.out is None:
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out_d =  '_'.join(("simple", args.metric, suffix))
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

    progress_file = os.path.join(out_d, 'mutations.json')
    if os.path.isdir(out_d):
        with open(progress_file, 'r') as f:
            files = json.load(f)
        results = np.array(files)
    else:
        os.mkdir(out_d)
        results = np.full(args.total, '', dtype = object)

    out_path = os.path.join(out_d, '{0!s}.json')
    enough_scenes = lambda l: len(np.argwhere(l == '')) == 0
    predicate = create_predicate(args.metric, args.threshold)

    materials = {'Wood' : 1.0}
    gen = ExpGen(materials, 'local', args.shape)
    phys = NoisyStability(noise = args.noise, frames = 240)

    params = {
        'base' : base,
        'size' : args.size,
        'gen'  : gen,
        'phys' : phys,
        'pred' : predicate,
        'out'  : out_path,
        'debug': False,
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
                client.cancel(ac.futures)

        files = results.tolist()
        with open(progress_file, 'w') as f:
            json.dump(files, f)

        client.close()

def initialize_dask(n, factor = 5, slurm = False):

    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)

    else:
        n = min(100, n)
        py = './enter_conda.sh python3'
        params = {
            'python' : py,
            'cores' : 1,
            'memory' : '512MB',
            'walltime' : '180',
            'processes' : 1,
            'job_extra' : [
                '--qos use-everything',
                # '--qos tenenbaum',
                '--array 0-{0:d}'.format(n - 1),
                '--requeue',
                '--output "/dev/null"'
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
