#!/usr/bin/env python3

""" Creates stimuli movies from image renderings.

Uses a timing file containing time points of interest
for each scene to be generated.

"""
import os
import json
import shlex
import argparse
import subprocess
import numpy as np
from pprint import pprint

from galileo_ramp.utils import config, encoders, ffmpeg
CONFIG = config.Config()

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('on_ramp', type = int,
                        help = 'Number of balls on ramp')
    parser.add_argument('renders', type = str,
                        help = 'Paths to the trial renderings')
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()

    workdir = CONFIG['PATHS', 'scenes']
    timing_file = '{0:d}_timings.json'.format(args.on_ramp)
    timing_file = os.path.join(workdir, timing_file)
    render_src = os.path.join(CONFIG['PATHS', 'renders'], args.renders)
    # Load timings
    with open(timing_file, 'r') as f:
        timings = json.load(f)

    # Set mask
    if args.mask:
        mask = os.path.join(CONFIG['PATHS', 'root'], 'galileo_ramp', 'world',
                            'render', 'Textures', 'mask.mp4')
    else:
        mask = None

    # Movies will be saved within the inference directory
    movie_dir = os.path.join(CONFIG['PATHS', 'movies'], 'movies_mask_{0:d}')
    movie_dir = movie_dir.format(args.mask)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    for trial_path, times in timings:

        parts = trial_path.split(os.sep)
        trial = '{0!s}_{1!s}'.format(parts[-2],
                                        parts[-1].replace('.json', ''))
        trial_path = os.path.join(render_src, trial)

        # Create motion component
        src_path = '{0!s}/render/%d.png'.format(trial_path)
        src_path = os.path.join(render_src, src_path)

        for cond, point in enumerate(times):
            out_path = 'trial_{0!s}_cond_{1:d}.mp4'.format(trial, cond)
            out_path = os.path.join(movie_dir, out_path)

            # Create raw video
            ffmpeg.ffmpeg(src_path, out_path, vframes = point)
            if not mask is None:
                ffmpeg.ffmpeg_concat(out_path, mask, base = 0)



if __name__ == '__main__':
    main()
