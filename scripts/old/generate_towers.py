#!/bin/python3
""" Generates towers for inspection.
"""

import os
import glob
import json
import argparse
from copy import deepcopy
from pprint import pprint

import numpy as np
import networkx as nx
import dask
from dask import distributed
from dask_jobqueue import SLURMCluster

from utils import plot, config
from blockworld import towers, blocks
from blockworld.simulation.generator import Generator


def initialize_dask(n, factor = 5, slurm = False):
    CONFIG = Config()
    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)

    else:
        n = min(100, max(1, int(n/factor)))
        cont = os.path.normpath(CONFIG['env'])
        bind = cont.split(os.sep)[1]
        bind = '-B /{0!s}:/{0!s}'.format(bind)
        py = 'singularity exec {0!s} {1!s} python3'.format(bind, cont)
        params = {
            'python' : py,
            'cores' : 1,
            'memory' : '1000MB',
            'walltime' : '180',
            'processes' : 1,
            'job_extra' : [
                '--qos use-everything',
                '--array 0-{0:d}'.format(n - 1),
                '--requeue'
            ],
            'env_extra' : [
                'JOB_ID=${SLURM_ARRAY_JOB_ID%;*}_${SLURM_ARRAY_TASK_ID%;*}',
                'source /etc/profile.d/modules.sh',
                'module add openmind/singularity/2.6.0']
        }
        cluster = SLURMCluster(**params)
        print(cluster.job_script())
        # cluster.scale(1)
        cluster.adapt(
            minimum = 1,
            maximum = n,
        )

    print(cluster.dashboard_link)
    return distributed.Client(cluster)

class MyGenerator(Generator):

    def __init__(self, materials, stability, block_size):
        self.materials = materials
        self.builder = stability
        self.block_size = np.array(block_size)

    def sample_blocks(self, n):
        """
        Procedurally generates blocks of cardinal orientations.
        """
        n = int(n)
        if n <= 0 :
            raise ValueError('n_blocks must be > 1.')
        for _ in range(n):
            block_dims = deepcopy(self.block_size)
            np.random.shuffle(block_dims)
            yield blocks.SimpleBlock(block_dims)

    def __call__(self, d):
        base = d['base']
        k = d['k']
        idx = d['idx']
        if not isinstance(base, towers.tower.Tower):
            try:
                base = list(base)
            except:
                raise ValueError('Unsupported base.')
            base = towers.EmptyTower(base)
        np.random.seed()
        new_tower = self.sample_tower(base, k)
        base_name = 'blocks_{0:d}_tower_{1:d}'
        base_name = base_name.format(len(new_tower), idx)
        return new_tower, base_name


def main():
    parser = argparse.ArgumentParser(
        description = 'Renders the towers in a given directory')
    parser.add_argument('n', type = int, help = 'Number of towers to generate')
    parser.add_argument('b', type = int, help = 'The size of each tower.')
    parser.add_argument('--base', type = int, nargs = 2, default = (2,1),
                        help = 'Dimensions of base.')
    parser.add_argument('--shape', type = int, nargs = 3, default = (2,1,1),
                        help = 'Dimensions of block (x,y,z).')
    parser.add_argument('--out', type = str, help = 'Path to save renders.',
                        default = 'towers')
    parser.add_argument('--base_path', type = str,
                        help = 'Path to base tower.')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')

    args = parser.parse_args()
    CONFIG = Config()
    out_d = os.path.join(CONFIG['data'], args.out)
    # out_d = args.out

    if args.base_path is None:
        base = args.base
        base_path = '{0:d}x{1:d}'.format(*base)
    else:
        base = towers.simple_tower.load(args.base_path)
        out_d += '_extended'
        base_path = os.path.basename(os.path.splitext(args.base_path)[0])

    if not os.path.isdir(out_d):
        os.mkdir(out_d)

    materials = {'Wood' : 1.0}
    gen = MyGenerator(materials, 'local', args.shape)
    tower_args = {'base' : base, 'k' : args.b}
    t = [{'idx': i, **tower_args} for i in range(args.n)]
    client = initialize_dask(args.n, slurm = args.slurm)
    # t = client.map(gen, t)
    # towers = client.gather(t)

    for future in distributed.as_completed(client.map(gen, t)):
        new_tower, base_name = future.result()
        base_name = '{0!s}_base_{1!s}.json'.format(base_name, base_path)
        out = os.path.join(out_d, base_name)
        with open(out, 'w') as f:
            json.dump(new_tower.serialize(), f, indent=4, sort_keys = True)

if __name__ == '__main__':
    main()
