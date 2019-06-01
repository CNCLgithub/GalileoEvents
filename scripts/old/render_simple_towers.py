import os
import sys
import json
import shlex
import argparse
import subprocess
import numpy as np
from pprint import pprint
from itertools import repeat

from slurmpy import sbatch
from dask import distributed

from blockworld.utils import json_encoders
from datasets.simple_dataset import SimpleDataset

from utils import config
CONFIG = sys.modules['utils.config'].Config()

dir_path = os.path.dirname(os.path.realpath(__file__))
render_path = os.path.join(dir_path, 'render.py')

mat_path = os.path.join(dir_path, 'materials.blend')
cmd = '/blender/blender -noaudio --background -P {0!s}'

# Originally imported from `stimuli.analyze_ke`.
# However issues with dask pickle forces me to put this here.

def render(scene_str, traces, out, mode = 'none'):
    """ Subprocess call to blender

    Arguments:
        scene_str (str): The serialized tower scene
        traces (dict): A collection of positions and orientations for each
                       block across time.
        theta (float): The camera angle in radians
        out (str): The directory to save renders
    """
    if not os.path.isdir(out):
        os.mkdir(out)
    t_path = os.path.join(out, 'trace.json')
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
        '--resolution',
        '512', '512',
        '--render_mode',
        mode,
        # '--theta',
        # '{0:f}'.format(theta),
    ]
    subprocess.run(_cmd)


def render_tower(src, trial, out, mode):
    """ Render tower with randomly sampled camera angle.

    For a given scene pair, render either the congruent or incongruent
    tower with the same camera angle.

    Call to this function preserves state. Will not re-sample camera angle
    or re-render completed frames.

    Arguments:
        src   (str): Path to dataset hdf5 file
        trial (int): Trial to render
        out   (str): Directory to save trial renderings

    Returns
        Nothing
    """
    out_render = os.path.join(out, '{0:d}'.format(trial))
    if not os.path.isdir(out_render):
        os.mkdir(out_render)

    dataset = SimpleDataset(src)
    tower, obs = dataset[trial]

    scene_str = json.dumps(tower.serialize())
    render(scene_str, [obs], out_render, mode)


def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', type = str,
                        help = 'Path to tower dataset')
    parser.add_argument('--run', type = str, default = 'local',
                        choices = ['batch', 'local'],
                        help = 'submission modes')
    parser.add_argument('--trial', type = int, nargs = '+',
                        help = 'Trial to render')
    parser.add_argument('--render_mode', type = str, default = 'none',
                        choices = ['none', 'frozen', 'motion', 'default'],
                        help = 'type of rendering')
    parser.add_argument('--batch', type = int, default = 10,
                        help = 'Size of sbatch array.')
    parser.add_argument('--out', type = str,
                        help = 'Path to render individual scene.')

    args = parser.parse_args()

    out = os.path.basename(os.path.splitext(args.src)[0])
    out = os.path.join(CONFIG['PATHS', 'renders'], out + '_render')
    if not os.path.isdir(out):
        os.mkdir(out)

    if args.trial is None:
        # render all trials is none are given
        dataset = SimpleDataset(args.src)
        trials = np.arange(len(dataset)).astype(int)
    else:
        trials = args.trial

    if args.run == 'batch':
        # submit `--batch` sbatch jobs to render trials.
        submit_sbatch(args.src, out, args.batch, trials, args.render_mode)

    else:
        # compute rendering
        client = create_client()
        futures = client.map(render_tower, repeat(args.src),
                             trials, repeat(out), repeat(args.render_mode),
                             resources = {'foo' : 1})
        results = client.gather(futures)
        client.close()

def submit_sbatch(src, out, chunks, trials, mode):
    """ Helper function that submits sbatch jobs.

    Arguments:
        src (str): Path to dataset
        out (str): Path to save trials
        trials (list): A list of trials to render
    """
    chunks = min(chunks, len(trials))

    interpreter = '#!/bin/bash'
    extras = [
        'source /etc/profile.d/modules.sh',
        'module add openmind/singularity/3.0',
        'export PYTHONPATH="{0!s}"'.format(CONFIG['PATHS', 'root']),
    ]
    resources = {
        'cpus-per-task' : '8',
        'mem-per-cpu' : '300MB',
        'time' : '1-0',
        'qos' : 'use-everything',
        'requeue' : None,
        'output' : os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out'),
        'exclude' : 'node[021,022]'
        # 'output' : '/dev/null'
    }
    flags = ['--run local', '--src {0!s}'.format(src),
             '--out ' + out, '--render_mode {}'.format(mode)]
    jobs = [('--trial {0:d}'.format(p),) for p in trials]
    path = os.path.realpath(__file__)
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './enter_conda.sh python3 {0!s}'.format(path)
    batch = sbatch.Batch(interpreter, func, jobs, flags, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=chunks)))
    batch.run(n = chunks, check_submission = False)

def create_client():
    """ Initializes a `dask.distributed` client for local computing.
    """
    cores =  len(os.sched_getaffinity(0))
    nworkers = int(np.ceil(cores / 8))
    cluster = distributed.LocalCluster(n_workers = nworkers,
                                       threads_per_worker = min(cores, 8),
                                       resources = {'foo' : nworkers}
    )
    print(cluster)
    print(cluster.dashboard_link)
    client = distributed.Client(cluster)
    return client


if __name__ == '__main__':
   main()

