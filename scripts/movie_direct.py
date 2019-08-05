#!/usr/bin/env python3

""" Creates movies from image renderings.

Creates a series of mp4s given timing points

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
    parser.add_argument('renders', type = str,
                        help = 'Paths to images')
    parser.add_argument('timings', type = int, nargs = '+',
                        help = 'Time points to cut video')
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()

    render_src = os.path.join(CONFIG['PATHS', 'renders'], args.renders)

    # Set mask
    if args.mask:
        mask = os.path.join(CONFIG['PATHS', 'root'], 'galileo_ramp', 'world',
                            'render', 'Textures', 'mask.mp4')
    else:
        mask = None

    movie_dir = os.path.join(CONFIG['PATHS', 'movies'], 'movies_mask_{0:d}')
    movie_dir = movie_dir.format(args.mask)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)


    # Create motion component
    src_path = '{0!s}/render/%04d.png'.format(render_src)
    src_path = os.path.join(render_src, src_path)

    for t in args.timings:
        out_path = args.renders.replace(os.sep, '_')
        out_path = os.path.join(movie_dir, '{0!s}_t-{1:d}.mp4'.format(out_path, t))

        # Create raw video
        ffmpeg.ffmpeg(src_path, out_path, vframes = t)
        if not mask is None:
            ffmpeg.ffmpeg_concat(out_path, mask, base = 0)



if __name__ == '__main__':
    main()
