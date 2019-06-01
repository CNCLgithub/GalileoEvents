import os
import copy
import json
import shlex
import pathlib
import tempfile
import argparse
import subprocess
from pprint import pprint

import dask
from dask import distributed
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import blockworld
from blockworld.utils import json_encoders
from blockworld.simulation import tower_scene, generator
from blockworld.simulation.substances import Substance

from config import Config
CONFIG = Config()
# from stimuli.analyze_ke import ExpGen, ExpStability

dir_path = os.path.dirname(os.path.realpath(__file__))
render_path = os.path.join(dir_path, 'render.py')

mat_path = os.path.join(dir_path, 'materials.blend')
cmd = '/blender/blender --background -P {0!s}'

# Originally imported from `stimuli.analyze_ke`.
# However issues with dask pickle forces me to put this here.

class ExpGen(generator.Generator):

    """
    Creates the configurations needed to evaluate critical blocks.
    """

    def mutate_block(self, tower, subs, apps, idx, mat):
        """ Helper that allows for indexed mutation.
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

def render(scene_str, traces, out):
    if not os.path.isdir(out):
        os.mkdir(out)
    t_path = os.path.join(out, 'trace.json')
    # with tempfile.NamedTemporaryFile(mode = 'w+', delete = False) as temp:
    with open(t_path, 'w') as temp:
        json.dump(traces, temp, cls = json_encoders.TowerEncoder)

    _cmd = cmd.format(render_path)
    _cmd = shlex.split(_cmd)
    _cmd += [
        '--',
        '--materials',
        mat_path,
        '--out',
        out,
        '--save_world',
        '--scene',
        scene_str,
        '--trace',
        t_path,
        '--render_mode',
        'none'
    ]
    subprocess.run(_cmd, check = True)

def simulate(tower):
    keys = list(tower.blocks.keys())[1:]
    with tower_scene.TowerPhysics(tower.serialize()) as scene:
        trace = scene.get_trace(120, keys, fps = 30)

    return trace

def render_quantile(tower_path, conf, out, mutate):
    # load and perturb tower
    if not os.path.isdir(out):
        os.mkdir(out)
    pert_path = os.path.join(out, os.path.basename(tower_path))
    original = blockworld.towers.simple_tower.load(tower_path)
    towers = [original]
    # if not os.path.isfile(pert_path):
    #     original = blockworld.towers.simple_tower.load(tower_path)
    #     perturb = ExpStability(noise = 0.10)
    #     # towers = perturb.perturb(original, n = 10)
    #     # towers[0] = original
    #     towers = [original]
    #     with open(pert_path, 'w') as f:
    #         towers_json = [tower.serialize() for tower in towers]
    #         json.dump(towers_json, f, indent=4, sort_keys=True)
    # else:
    #     with open(pert_path, 'r') as f:
    #         towers = json.load(f)
    #     towers = [blockworld.towers.simple_tower.load(t) for t in towers]

    # retrieve configuration
    subs = towers[0].extract_feature('substance')
    apps = towers[0].extract_feature('appearance')
    block_idx, mat = conf.split('_')
    block_idx = int(block_idx) - 1

    if not mutate:
        # render default tower
        traces = list(map(simulate, towers))
        out_render = os.path.join(out, 'template')
        scene_str = json.dumps(towers[0].serialize())
        render(scene_str, traces, out_render)

    else:
        # render mutation
        gen = ExpGen({'Wood' : 1.0}, 'local', )
        mutated = []
        for i in range(len(towers)):
            t = gen.mutate_block(towers[i], subs, apps, block_idx, mat)
            mutated.append(t)

        traces = list(map(simulate, mutated))
        out_render = os.path.join(out, conf)
        scene_str = json.dumps(mutated[0].serialize())
        render(scene_str, traces, out_render)

def create_client(n = 12, slurm = False):

    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)
    else:
        cont = os.path.normpath(CONFIG['env'])
        chunk = n
        bind = cont.split(os.sep)[1]
        bind = '-B /{0!s}:/{0!s}'.format(bind)
        py = 'singularity exec {0!s} {1!s} python3'.format(bind, cont)
        params = {
            'python' : py,
            'cores' : 4,
            'memory' : '2000MB',
            'processes' : 1,
            'walltime' : '80',
            'job_extra' : [
                '--qos use-everything',
                # '--array 0-{0:d}'.format(chunk - 1),
                '--requeue',
            ],
            'env_extra' : [
                # 'JOB_ID=${SLURM_ARRAY_JOB_ID%;*}_${SLURM_ARRAY_TASK_ID%;*}',
                'source /etc/profile.d/modules.sh',
                'module add openmind/singularity/2.6.0',
                'export PYTHONPATH="{0!s}"'.format(dir_path),
            ],
            'local_directory' : '$TMPDIR',
        }
        cluster = SLURMCluster(**params)
        print(cluster.job_script())
        # cluster.scale(1)
        cluster.adapt(
            minimum = n,
            maximum = n,
        )

    print(cluster)
    client = distributed.Client(cluster)
    print(cluster.dashboard_link)
    return client

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')

    args = parser.parse_args()

    src = os.path.join(CONFIG['data'], args.src + '_quantiles.json')
    render_out = os.path.join(CONFIG['data'], args.src + '_diff_render')
    if not os.path.isdir(render_out):
        os.mkdir(render_out)

    # Get quantiles for each dimension
    with open(src, 'r') as f:
        quantiles = json.load(f)

    client = create_client(n = len(quantiles) * 2,
                           slurm = args.slurm)
    futures = []
    for q in quantiles:
        template_path = q
        out = q['dim'] + '_' + q['quant']
        out = os.path.join(render_out, out)
        for i in [True, False]:
            future = client.submit(render_quantile, q['path'], q['id'], out, i)
            futures.append(future)

    for future in distributed.as_completed(futures):
        not_done = list(map(lambda x : not x.done(), futures))
        client.rebalance(not_done)
    client.profile()

if __name__ == '__main__':
   main()
