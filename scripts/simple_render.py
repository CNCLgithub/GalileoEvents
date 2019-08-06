#!/usr/bin/env python
import os
import sys
import glob
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

def render_scene(src, out, res, mode, snapshot = False):
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
        out = out,
        render_mode = mode,
        resolution = res,
        theta = 45,
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
                        help = 'Path to scene json')
    parser.add_argument('--run', type = str, default = 'local',
                        choices = ['batch', 'local'],
                        help = 'submission modes')
    parser.add_argument('--mode', type = str, default = 'none',
                        choices = ['default', 'none',],
                        help = 'rendering mode.')
    parser.add_argument('--out', type = str,
                        help = 'Path to render individual scene.')
    parser.add_argument('--res', type = int, nargs = 2, default = (512,512),
                        help = 'Resolution for images')

    args = parser.parse_args()


    if args.out is None:
        out = '{0!s}_rendered'.format(os.path.dirname(args.src))
    else:
        out = os.path.join(CONFIG['PATHS', 'renders'], args.out)

    if not os.path.isdir(out):
        os.makedirs(out)

    render_scene(args.src, out, args.res, args.mode)


if __name__ == '__main__':
    main()
