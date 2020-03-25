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

from physics.utils import ffmpeg
from galileo_ramp.exp1_dataset import Exp1Dataset

def noise_mask(src, out, dur, fps):
    """ Creates white noise mask """
    cmd = 'ffmpeg -y -f lavfi -r {0:d} -i nullsrc=s=600x400 -filter_complex "geq=random(1)*255:128:128" -t {1:f} -pix_fmt yuv420p {2!s}'
    cmd = cmd.format(fps, dur, out)
    return cmd

def black_mask(src, out, dur, fps):
    cmd = 'ffmpeg -y -f lavfi -r {0:d} -i color=black:600x400:d={1:f} -pix_fmt yuv420p {2!s}'
    cmd = cmd.format(fps, dur, out)
    return cmd


def stimuli_with_mask(src, fps, dur, out):
    #cmds = ffmpeg.chain([black_mask, ffmpeg.concat],
    #                    [(dur, fps), (src, False)],
    #                    src, out, 'e')
    cmds = ffmpeg.chain([black_mask, ffmpeg.concat],
                        [(dur, fps), (src, False)],
                        src, out, 'e')
    ffmpeg.run_cmd(cmds)

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset', type = str,
                        help = 'Path to dataset')
    args = parser.parse_args()

    dataset = Exp1Dataset(args.dataset)


    # Movies will be saved within the inference directory
    base_path = os.path.basename(args.dataset)
    base_path = os.path.splitext(base_path)[0]
    movie_dir = '/movies/{0!s}'.format(base_path)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    render_path = os.path.join('/renders', base_path)

    for i in range(len(dataset)):
    #for i in range(1):
        _, _, timings = dataset[i]
        src_path = os.path.join(render_path, str(i), 'render',
                                '%d.png')
        for cond, point in enumerate(timings):
            out_path = '{0:d}_t-{1:d}'.format(i, cond)
            out_path = os.path.join(movie_dir, out_path)


            # Create continous video
            out_cont = out_path + '_continous'
            ffmpeg.continous_movie(src_path, out_cont,
                                   vframes = point)

            # add mask
            dur = 0.250
            stimuli_with_mask(out_cont + '.mp4', 60, dur, out_path)



if __name__ == '__main__':
    main()
