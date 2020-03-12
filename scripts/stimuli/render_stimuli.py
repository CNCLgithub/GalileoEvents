#!/usr/bin/env python

""" Renders a set of scenes for use as stimuli"""

import os
import sys
import json
import h5py
import argparse
import subprocess
import numpy as np
from pprint import pprint
from itertools import repeat

from slurmpy import sbatch

from physics.world.render.interface import render

from galileo_ramp.exp1_dataset import Exp1Dataset

base_path = '/project/deps/ramp_physics/physics/world/render/'
render_path = base_path + 'render.py'
scene_path = '/project/galileo_ramp/blend/new_scene.blend'

def render_trace(scene, trace, out, res, mode,
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

    # Render
    kwargs = dict(
        scene = {'scene': scene},
        trace = trace,
        out = out,
        render_mode = mode,
        resolution = res,
        theta = 1.5*np.pi,
        render = render_path,
        materials = scene_path,
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
    parser.add_argument('src', type = str,
                        help = 'Path to scenes')
    parser.add_argument('--idx', type = int,
                        help = 'scene idx')
    parser.add_argument('--resolution', type = int, nargs = 2,
                        default = (600, 400),
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

    args = parser.parse_args()

    out = os.path.basename(args.src)
    out = os.path.splitext(out)[0]
    out = os.path.join('/project','output','renders', out)
    if not os.path.isdir(out):
        os.mkdir(out)

    if args.run == 'batch':
        # submit `--batch` sbatch jobs to render trials.
        submit_sbatch(args)
    else:
        dataset = Exp1Dataset(args.src)
        if not args.idx is None:
            scene_out = os.path.join(out, str(args.idx))
            scene, trace, _ = dataset[args.idx]
            print(scene)
            render_trace(scene, trace, scene_out, args.resolution,
                            args.mode, args.snapshot, args.gpu)


        else:
            for idx, scene in enumerate(dataset):
                scene_out = os.path.join(out, str(idx))
                scene, trace, _ = dataset[idx]
                render_trace(scene, trace, scene_out, args.resolution,
                             args.mode, args.snapshot, args.gpu)

def submit_sbatch(args, chunks = 210):
    """ Helper function that submits sbatch jobs.

    Arguments:
        src (str): Path to dataset
        out (str): Path to save trials
        trials (list): A list of trials to render
    """
    dataset = Exp1Dataset(args.src)

    njobs = min(chunks, len(dataset))

    tasks = [(args.src,'--idx ' +str(i))
             for i in range(len(dataset))]
    kwargs = ['--run local',
              '--mode {0!s}'.format(args.mode)]
    if args.snapshot:
        kwargs += ['--snapshot',]
    if args.gpu:
        kwargs += ['--gpu',]

    interpreter = '#!/bin/bash'
    extras = []
    resources = {
        'cpus-per-task' : '8',
        'mem-per-cpu' : '300MB',
        'time' : '200',
        'partition' : 'scavenge',
        'requeue' : None,
        # 'output' : '/dev/null',
    }
    path = os.path.realpath(__file__)
    path = os.path.join('/project/scripts/stimuli', os.path.basename(path))
    func = 'bash {0!s}/run.sh {1!s}'.format(os.getcwd(), path)
    batch = sbatch.Batch(interpreter, func, tasks, kwargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    batch.run(n = njobs, check_submission = True)

if __name__ == '__main__':
   main()
