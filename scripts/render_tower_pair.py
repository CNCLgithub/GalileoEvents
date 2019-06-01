import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pprint import pprint
from itertools import repeat

from slurmpy import sbatch

from blockworld.utils import json_encoders
from experiment.render.interface import render
from experiment.dataset.particle_dataset import ParticleDataset

from utils import config
CONFIG = sys.modules['utils.config'].Config()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_2vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_angle(pos, t = 1E-2):
    xy_dir = pos[-1, :, :2] - pos[0, :, :2]
    xy_dir = np.mean(xy_dir, axis = 0)
    print(xy_dir)
    if np.mean(xy_dir) < t:
        xy_dir = (1, 0)
    return angle_2vec((1,0), xy_dir), xy_dir


def render_tower(src, trial, out, mode, snapshot = False):
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

    dataset = ParticleDataset(src)
    (obs, block_ids), tower = dataset[trial]
    if trial % 2 == 0:
        pair_idx = trial + 1
    else:
        pair_idx = trial - 1

    (obs2, _), t2 = dataset[pair_idx]
    theta_dir = os.path.join(out,'{0:d}'.format(min(trial, pair_idx)))
    theta_file = os.path.join(theta_dir, 'theta.json')

    # Only look at the trajectories if the unknown blocks
    idxs = np.array(block_ids) - 1
    # Compute the angle between the averaged trajectories
    theta_1, xy_dir_1 = compute_angle(obs['position'][:, idxs])
    theta_2, xy_dir_2 = compute_angle(obs2['position'][:, idxs])
    theta_delta = angle_2vec(xy_dir_1, xy_dir_2)
    print(xy_dir_1, xy_dir_2)
    print(theta_delta, theta_1, theta_2)
    # Add a small amount of noise
    noise = np.random.normal(scale=np.pi * (3.0 / 180.0))
    reference = theta_1 + theta_delta / 2.
    # If the intersitial angle is small, place camera to the side
    # Otherwise, the camera will be in between the towers falling
    if abs(theta_delta) < (np.pi / 3):
        if np.random.randint(2):
            reference += np.pi/3
        else:
            reference -= np.pi/3
    theta = reference + noise

    # Use a previously saved theta if exists
    if os.path.isfile(theta_file):
        with open(theta_file, 'r') as f:
            theta = json.load(f)['theta']
    # Save theta in case of interruptions
    else:
        if not os.path.isdir(theta_dir):
            os.mkdir(theta_dir)
        with open(theta_file, 'w') as f:
            json.dump({'theta' : theta}, f)

    # Render
    scene_str = json.dumps(tower.serialize())
    kwargs = dict(
        scene = scene_str,
        traces = [obs],
        theta = theta,
        out = out_render,
        render_mode = mode,
        blocks = block_ids,
        resolution = ('512', '512'),
    )
    if snapshot:
        kwargs['frames'] = [0]
    render(**kwargs)

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
    parser.add_argument('--mode', type = str, default = 'none',
                        choices = ['default', 'none', 'motion', 'frozen',])
    parser.add_argument('--trial', type = int, nargs = '+',
                        help = 'Trial to render')
    parser.add_argument('--snapshot', action = 'store_true',
                        help = 'Only render first frame of each tower')
    parser.add_argument('--batch', type = int, default = 10,
                        help = 'Size of sbatch array.')
    parser.add_argument('--out', type = str,
                        help = 'Path to render individual scene.')

    args = parser.parse_args()

    src = os.path.join(CONFIG['PATHS', 'databases'], args.src)
    out = os.path.basename(os.path.splitext(args.src)[0])
    out = os.path.join(CONFIG['PATHS', 'renders'], out + '_render')
    if not os.path.isdir(out):
        os.mkdir(out)

    if args.trial is None:
        # render all trials is none are given
        dataset = ParticleDataset(src)
        trials = np.arange(len(dataset)).astype(int)
        if args.snapshot:
            trials = np.arange(len(dataset), step = 2).astype(int)
    else:
        trials = args.trial

    if args.run == 'batch':
        # submit `--batch` sbatch jobs to render trials.
        submit_sbatch(src, out, args.batch, trials, args.snapshot,
                      args.mode)

    else:
        # compute rendering
        for t in trials:
            render_tower(src, t, out, args.mode, args.snapshot)

def submit_sbatch(src, out, chunks, trials, snapshot, mode):
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
        'cpus-per-task' : '24',
        'mem-per-cpu' : '2GB',
        'time' : '1-0',
        # 'qos' : 'use-everything',
        'qos' : 'tenenbaum',
        'requeue' : None,
        'output' : os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out'),
        # 'output' : '/dev/null',
        'exclude' : 'node[021,022]',
    }
    flags = ['--run local', '--src {0!s}'.format(src),
             '--mode {}'.format(mode),
             '--out ' + out]
    if snapshot:
        flags += ['--snapshot',]
    jobs = [('--trial {0:d}'.format(p),) for p in trials]
    path = os.path.realpath(__file__)
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './run.sh python3 {0!s}'.format(path)
    batch = sbatch.Batch(interpreter, func, jobs, flags, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=chunks)))
    batch.run(n = chunks, check_submission = False)

if __name__ == '__main__':
   main()

