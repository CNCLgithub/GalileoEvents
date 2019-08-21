#!/usr/bin/env python3

""" Creates stimuli movies from image renderings.

Uses a timing file containing time points of interest
for each scene to be generated.

"""
import os
import json
import argparse
import numpy as np

from galileo_ramp.utils import config, encoders
CONFIG = config.Config()

def get_colors(scene_file):
    """ Returns the unique list of colors from ramp -> table
    """
    with open(scene_file, 'r') as f:
        scene = json.load(f)
    objects = scene['scene']['objects']
    n_objs = len(objects)
    colors = [objects[str(k)]['appearance'] for k in range(n_objs)]
    colors = np.array(colors)
    _, idx = np.unique(colors, return_index=True)
    return colors[np.sort(idx)].tolist()

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('on_ramp', type = int, nargs = '+',
                        help = 'Number of balls on ramp')
    parser.add_argument('--n_cond', type = int, default = 5,
                        help = 'Number of conditions')
    parser.add_argument('movies', type = str,
                        help = 'Paths to the trial renderings')
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()


    workdir = CONFIG['PATHS', 'scenes']

    trial_sources = []
    for n in args.on_ramp:
        timing_file = '{0:d}_timings.json'.format(n)
        timing_file = os.path.join(workdir, timing_file)
        with open(timing_file, 'r') as f:
            data = json.load(f)
            trial_sources += list(zip(*data))[0]

    colors = list(map(get_colors, trial_sources))

    condition_list = []
    i = 0
    for cond in range(args.n_cond):

        clist = []
        for t, (src, color) in enumerate(zip(trial_sources, colors)):
            parts = src.split(os.sep)
            trial = '{0!s}_{1!s}'.format(parts[-2],
                                         parts[-1].replace('.json', ''))
            idx = int((i + cond) % args.n_cond)
            path = 'trial_{0!s}_cond_{1:d}.mp4'.format(trial, idx)
            clist.append((path, color))
            i += 1

        condition_list.append(clist)

    out_path = os.path.join(workdir, 'condlist.json')
    with open(out_path, 'w') as f:
        json.dump(condition_list, f, indent = 2)

    msg = ('Wrote list with {0:d} conditions and {1:d}' + \
        ' trials per condition.').format(args.n_cond, len(trial_sources))
    print(msg)


if __name__ == '__main__':
    main()
