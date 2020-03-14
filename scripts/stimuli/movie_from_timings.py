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

def stimuli_with_mask(src, fps, dur, out):
    cmds = ffmpeg.chain([noise_mask, ffmpeg.concat],
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
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()

    dataset = Exp1Dataset(args.dataset)


    # Movies will be saved within the inference directory
    base_path = os.path.basename(args.dataset)
    base_path = os.path.splitext(base_path)[0]
    movie_dir = '/movies/{0!s}'.format(base_path)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    render_path = os.path.join('/renders', base_path)

    # for i in range(len(dataset)):
    for i in [0]:
        _, _, timings = dataset[i]
        col = timings[0]
        # -66ms -> +266ms @ 60fps
        scale = 6
        times = np.array([-1, 1, 2, 3]) * scale
        times += col

        src_path = os.path.join(render_path, str(i), 'render',
                                '%d.png')

        for cond, point in enumerate(times):
            out_path = '{0:d}_cond_{1:d}'.format(i, cond)
            out_path = os.path.join(movie_dir, out_path)


            # Create continous video
            out_cont = out_path + '_continous'
            ffmpeg.continous_movie(src_path, out_cont,
                                   vframes = point)

            # add mask
            out_mask = out_path + '_mask'
            dur = 2.0 - (point / 60.0)
            stimuli_with_mask(out_cont + '.mp4', 60, dur, out_mask)



if __name__ == '__main__':
    main()
