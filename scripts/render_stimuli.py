#!/usr/bin/env python

""" Renders a set of scenes for use as stimuli"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pprint import pprint
from itertools import repeat

from slurmpy import sbatch

from galileo_ramp.utils import config
from galileo_ramp.world.render.interface import render
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def render_scene(src, out, res, mode,
                 snapshot = False, gpu = False):
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
    parts = src.split(os.sep)
    out_path = '{0!s}_{1!s}'.format(parts[-2],
                                    parts[-1].replace('.json', ''))
    out_path = os.path.join(out, out_path)

    with open(src, 'r') as f:
        data = json.load(f)

    scene = data['scene']

    if 'trace' in data:
        trace = data['trace']
    else:
        trace = forward_model.simulate(data['scene'], 900)
        trace = dict(zip(['pos', 'orn', 'lvl', 'avl', 'col'], trace))
    # Render
    kwargs = dict(
        scene = json.dumps(scene,),
        trace = trace,
        out = out_path,
        render_mode = mode,
        resolution = res,
        theta = 1.5*np.pi,
    )
    if snapshot:
        kwargs['frames'] = [0]
    if gpu:
        kwargs['gpu'] = None
    render(**kwargs)

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str, nargs = '+',
                        help = 'Path to scenes')
    parser.add_argument('--resolution', type = int, nargs = 2,
                        default = (1280,720),
                        help = 'Resolution for images')
    parser.add_argument('--run', type = str, default = 'local',
                        choices = ['batch', 'local'],
                        help = 'submission modes')
    parser.add_argument('--mode', type = str, default = 'none',
                        choices = ['default', 'none',],
                        help = 'rendering mode.')
    parser.add_argument('--snapshot', action = 'store_true',
                        help = 'Only render first frame of each scene')
    parser.add_argument('--batch', type = int, default = 10,
                        help = 'Size of sbatch array.')
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Use CUDA rendering')
    parser.add_argument('--out', type = str, default = 'stimuli',
                        help = 'Path to render individual scene.')

    args = parser.parse_args()

    out = os.path.join(CONFIG['PATHS', 'renders'], args.out)
    if not os.path.isdir(out):
        os.mkdir(out)


    if args.run == 'batch':
        # submit `--batch` sbatch jobs to render trials.
        submit_sbatch(args, out)
    else:
        # compute rendering
        for src in args.src:
            render_scene(src, out, args.resolution, args.mode,
                         args.snapshot, args.gpu)

def submit_sbatch(args, out, chunks = 100):
    """ Helper function that submits sbatch jobs.

    Arguments:
        src (str): Path to dataset
        out (str): Path to save trials
        trials (list): A list of trials to render
    """
    chunks = min(chunks, len(args.src))

    interpreter = '#!/bin/bash'
    extras = []
    resources = {
        'cpus-per-task' : '8',
        'mem-per-cpu' : '2GB',
        'time' : '1-0',
        'qos' : 'use-everything',
        'requeue' : None,
        'output' : os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out'),
        # 'output' : '/dev/null',
    }
    flags = ['--run local',
             '--mode {0!s}'.format(args.mode),
             '--out {0!s}'.format(out)]
    if args.snapshot:
        flags += ['--snapshot',]
    if args.gpu:
        flags += ['--gpu',]
    jobs = [(f,) for f in args.src]
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
