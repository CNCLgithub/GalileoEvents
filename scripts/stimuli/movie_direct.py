#!/usr/bin/env python3

""" Creates movies from image renderings.

Creates a series of mp4s given timing points

"""
import os
import argparse
import numpy as np

from physics.utils import ffmpeg

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('renders', type = str,
                        help = 'Paths to images')
    parser.add_argument('--timings', type = int, nargs = '+',
                        default = [120],
                        help = 'Time points to cut video')
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()

    # Set mask
    mase = None
    # if args.mask:
    #     # mask = os.path.join(CONFIG['PATHS', 'root'], 'galileo_ramp', 'world',
    #     #                     'render', 'Textures', 'mask.mp4')
    # else:
    #     mask = None

    movie_dir = '/movies/exp1_mask_{0:d}'
    movie_dir = movie_dir.format(args.mask)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)


    # Create motion component
    src_path = '{0!s}/render/%d.png'.format(args.renders)

    for t in args.timings:
        out_path = os.path.basename(args.renders)
        out_path = '{0!s}_t-{1:d}.mp4'.format(out_path, t)
        out_path = os.path.join(movie_dir, out_path)

        # Create raw video
        ffmpeg.continous_movie(src_path, out_path, fps = 60,
                               vframes = t)


if __name__ == '__main__':
    main()
