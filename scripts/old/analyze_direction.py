#!/bin/python3
""" Computes physical statistics over block towers.
"""

import os
import glob
import copy
import json
import argparse
from pprint import pprint
from copy import deepcopy
from itertools import repeat


import numpy as np
import pandas as pd
import dask
from dask import distributed
from dask_jobqueue import SLURMCluster

from utils import config
from blockworld import towers, blocks
from blockworld.simulation import physics, generator, tower_scene
from blockworld.simulation.substances import Substance
from experiment.hypothesis.block_hypothesis import simulate


CONFIG = config.Config()


class ExpGen(generator.Generator):

    """
    Creates the configurations needed to evaluate critical blocks.
    """

    def __init__(self, materials, stability, block_size):
        self.materials = materials
        self.builder = stability
        self.block_size = np.array(block_size)

    def mutate_block(self, tower, subs, apps, idx, mat):
        """ Helper that allows  for indexed mutation.
        """
        mt = copy.deepcopy(subs)
        mt[idx] = Substance(mat).serialize()
        app = copy.deepcopy(apps)
        app[idx] = mat
        base = tower.apply_feature('appearance', app)
        return base.apply_feature('substance', mt)

    def configurations(self, tower, others = None):
        """
        Generator for different tower configurations.
        Arguments:
            tower (`dict`) : Serialized tower structure.
        Returns:
            A generator with the i-th iteration representing the i-th
            block in the tower being replaced.
            Each iteration contains a dictionary of tuples corresponding
            to a tower with the replaced block having congruent or incongruent
            substance to its appearance, organized with respect to congruent
            material.
            { 'mat_i' : [block_i,...,]
              ...
        """
        if others is None:
            others = self.unknowns
        subs = tower.extract_feature('substance')
        apps = tower.extract_feature('appearance')
        for block_i in range(len(tower)):
            d = {mat : self.mutate_block(tower, subs, apps, block_i, mat)
                 for mat in others}
            yield d

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

    def __call__(self, base, n_blocks):
        if not isinstance(base, towers.tower.Tower):
            try:
                base = list(base)
            except:
                raise ValueError('Unsupported base.')
            base = towers.EmptyTower(base)
        np.random.seed()
        return self.sample_tower(base, n_blocks)

class ExpStability(physics.TowerEntropy):

    """
    Inherits `physic.TowerEntropy` to compute direction and
    stability information over a given tower.
    """

    def simulate(self, tower):
        """
        Controls simulations and extracts trace
        """
        return simulate(tower, self.frames)


    def direction(self, initial, final):
        """ Determines the direction that the tower falls.

        Defined in Battalgia 2013 as the average final
        positions of all blocks in the xy plane.

        Adapted from David Wolever @ https://bit.ly/2rQe1Fu

        Arguments:
           final (np.ndarray): Final positions of each block.

        Returns:
           (angle, mag) : A tuple containing the angle of falling
        and its magnitude.
        """
        vec = np.mean((final - initial)[:, :2], axis = 0)
        angle = angle_2vec((1,0), vec)
        mag = np.linalg.norm(vec)
        return angle, mag


    def stability(self, started, ended):
        """ Evaulates the stability of the tower.

        Defined in Battalgia 2013 as the proportion of
        blocks who have had a change in the z axis > 0.25
        """
        delta_z = abs(ended[:, 2] - started[:, 2])
        fall_ratio = np.sum(delta_z > 0.25) / len(delta_z)
        return fall_ratio

    def kinetic_energy(self, tower, positions):
        """ Computes KE
        """
        vel = physics.velocity(positions)
        vel = np.linalg.norm(vel, axis = -1)
        phys_params = tower.extract_feature('substance')
        density  = np.array([d['density'] for d in phys_params])
        volume = np.array([np.prod(tower.blocks[i+1]['block'].dimensions)
                           for i in range(len(tower))])
        mass = np.expand_dims(density * volume, axis = -1)
        # sum the vel^2 for each object across frames
        ke = 0.5 * np.dot(np.square(vel), mass).flatten()
        return np.sum(ke)

    def analyze(self, tower):
        """ Returns the stability statistics for a given tower.
        Args:
          tower : a `blockworld.towers.Tower`.
        Returns:
          A `dict` containing several statistics.
        """
        # original configuration
        trace = self.simulate(tower)
        pos = trace['position']
        angle, mag = self.direction(pos[0], pos[-1])
        stability = self.stability(pos[0], pos[-1])
        return {
            'angle' : angle,
            'mag' : mag,
            'instability' : stability,
            'trace' : trace,
        }

    def __call__(self, tower, configurations = None):
        """
        Evaluates the stability of the tower at each block.
        Returns:
          - The randomly sampled congruent tower.
          - The stability results for each block in the tower.
        """
        d = [{'id' : 'template', **self.analyze(tower)}]

        if not configurations is None:
            for b_id, conf in enumerate(configurations):
                mats = list(conf.keys())
                mat_towers = list(conf.values())
                kes = list(map(self.analyze, mat_towers))
                for idx, m in enumerate(mats):
                    id_str = '{0:d}_{1!s}'.format(b_id + 1, m)
                    d.append({'id' : id_str, **kes[idx]})
        return d

def simulate_tower(tower_path):
    # Load towers
    basename = os.path.basename(tower_path)
    name = os.path.splitext(basename)[0].split('_orig')[0]
    mut_path = glob.glob(tower_path.replace('orig', '*_*'))
    original = towers.simple_tower.load(tower_path)
    mutated = [towers.simple_tower.load(p) for p in mut_path]
    mut_names = [os.path.splitext(s)[0].split(name + '_')[-1] for
                 s in mut_path]

    # setup physics and configurations
    phys = ExpStability(noise = 0.2, frames = 120)
    materials = {'Wood' : 1.0}
    gen = ExpGen(materials, 'local')
    # compute statistics
    stats = {'template' : phys.analyze(original)}
    # stats = [{'tower': name, 'id' : 'template',
    #           **phys.analyze(original)}]
    for m, label in zip(mutated, mut_names):
        stats.update({label : phys.analyze(m)})
        # stats.append({'tower' : name, 'id': label,
        #               **phys.analyze(m)})
    return {name : stats}
    # return stats

def main():
    parser = argparse.ArgumentParser(
        description = 'Renders the towers in a given directory')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')
    parser.add_argument('--search', action = 'store_true',
                        help = 'Search through tower configurations.')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')

    args = parser.parse_args()

    src = os.path.join(CONFIG['data'], args.src)
    tower_jsons = glob.glob(os.path.join(src, '*_orig.json'))

    client = initialize_dask(len(tower_jsons), slurm = args.slurm)
    # results = []
    results = {}
    for f in distributed.as_completed(client.map(simulate_tower, tower_jsons)):
        # results.append(f.result())
        results.update(f.result())

    out = src + '_stability.json'
    with open(out, 'w') as f:
        json.dump(results, f)

def initialize_dask(n, factor = 1, slurm = False):

    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)

    else:
        n = min(100, max(1, int(n/factor)))
        chunk = 10
        cont = os.path.normpath(CONFIG['env'])
        bind = cont.split(os.sep)[1]
        bind = '-B /{0!s}:/{0!s}'.format(bind)
        py = 'singularity exec {0!s} {1!s} python3'.format(bind, cont)
        params = {
            'python' : py,
            'cores' : 1,
            'memory' : '500MB',
            'walltime' : '120',
            'processes' : 1,
            'job_extra' : [
                '--qos use-everything',
                '--array 0-{0:d}'.format(chunk - 1),
                '--requeue',
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
            minimum = n,
            maximum = n,
        )

    print(cluster.dashboard_link)
    client = distributed.Client(cluster)
    return client

if __name__ == '__main__':
    main()

