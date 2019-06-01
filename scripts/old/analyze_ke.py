#!/bin/python3
""" Computes physical statistics over block towers.
"""

import os
import glob
import copy
import json
import argparse
from pprint import pprint
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

import pybullet as p
# import logging
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

CONFIG = config.Config()

class ExpGen(generator.Generator):

    """
    Creates the configurations needed to evaluate critical blocks.
    """

    def simulate(self, tower):
        """
        Controls simulations and extracts positions
        """
        tower_s = tower.serialize()
        keys = list(tower.blocks.keys())[1:]
        with tower_scene.TowerPhysics(tower_s) as scene:
            trace = scene.get_trace(self.frames, keys)
        return trace['position']

    def mutate_block(self, tower, subs, apps, idx, mat):
        """ Helper that allows  for indexed mutation.
        """
        mt = copy.deepcopy(subs)
        mt[idx] = Substance(mat).serialize()
        app = copy.deepcopy(apps)
        app[idx] = mat
        base = tower.apply_feature('appearance', app)
        return base.apply_feature('substance', mt)

    def configurations(self, tower):
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
        subs = tower.extract_feature('substance')
        apps = tower.extract_feature('appearance')
        for block_i in range(len(tower)):
            d = {mat : self.mutate_block(tower, subs, apps, block_i, mat)
                 for mat in self.unknowns}
            yield d


class ExpStability(physics.TowerEntropy):

    """
    Inherits `physic.TowerEntropy` to compute direction and
    stability information over a given tower.
    """


    def direction(self, vel):
        """ Determines the direction that the tower falls.

        Adapted from David Wolever @ https://bit.ly/2rQe1Fu
        """
        vel = np.mean(vel[0, :, :2], axis = 0)
        # vel_u = vel / np.linalg.norm(vel)
        # ref = np.array([1,0])
        # radians = np.arccos(np.clip(np.dot(ref, vel_u), -1., 1.))
        return vel


    def stability(self, vel):
        """ Implements the Geometric stability metric from Hamrick 2016.
        """
        vel = np.linalg.norm(vel[0], axis = 1)[:-1]
        fall_ratio = np.sum(vel > 1E-3) / len(vel)
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
        orig_pos = self.simulate(tower)
        orig_vels = physics.velocity(orig_pos[[0, -1], :])
        orig_dir = self.direction(orig_vels)
        orig_stability = self.stability(orig_vels)
        d = {'orig_direction_x' : orig_dir[0],
             'orig_direction_y' : orig_dir[1],
             'orig_instability' : orig_stability}

        # perturbations = self.perturb(tower)
        # positions = list(map(self.simulate, perturbations))
        # vel_f = lambda array : physics.velocity(array[[0,-1], :])
        # vels = list(map(vel_f, positions))
        # directions = list(map(self.direction, vels))
        # stabilities = list(map(self.stability, vels))
        # ke_f = lambda array : self.kinetic_energy(tower, array)
        # kes = list(map(ke_f, positions))
        # directions = np.mean(directions, axis = 0)
        # d.update({ 'direction_x' : directions[0],
        #                   'direction_y' : directions[1],
        #                   'instability' : np.mean(stabilities),
        #                   'ke' : np.mean(kes)})
        return d

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



def simulate_tower(tower_j, search = False):
    # setup physics and configurations
    phys = ExpStability(noise = 0.2, frames = 90)
    materials = {'Wood' : 1.0}
    gen = ExpGen(materials, 'local')
    tower_name = os.path.splitext(os.path.basename(tower_j))[0]
    t = towers.simple_tower.load(tower_j)
    if search:
        others = gen.configurations(t)
    else:
        others = None
    # compute statistics
    stats = phys(t, configurations = others)
    stats = [{'tower' : tower_name, **k} for k in stats]
    return stats


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
    out = ''
    tower_jsons = glob.glob(os.path.join(src, '*.json'))

    client = initialize_dask(len(tower_jsons), slurm = args.slurm)
    results = []
    for f in distributed.as_completed(client.map(simulate_tower, tower_jsons, repeat(args.search))):
        results += f.result()

    out = src + '_stability.json'
    with open(out, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()

