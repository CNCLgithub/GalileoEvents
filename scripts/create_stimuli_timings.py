#!/usr/bin/env python
""" Creates scene timings based off collision data
"""

import os
import json
import glob
import string
import argparse
import numpy as np

from galileo_ramp.utils import config, encoders
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def compute_timings(ramp_file, before, after, dur = 900):
    """ Returns a tuple describing 5 temporal conditions

    Time in frames
    """
    with open(ramp_file, 'r') as f:
        data = json.load(f)
    state = forward_model.simulate(data['scene'], dur)
    collided = np.sum(state[-1], axis = -1) > 0
    collided = np.flatnonzero(collided).astype(int)
    stopped = np.sum(np.abs(state[3]), axis = (2, 1)) > 1e-4
    stopped = int(np.flatnonzero(stopped)[-1])
    return (collided[0] - before, collided[0] + after,
            collided[1] - before, collided[1] + after, stopped)

def main():

    parser = argparse.ArgumentParser(
        description = 'Creates scene timings based off collision data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('on_ramp', type = int,
                        help = 'Number of balls on ramp')
    parser.add_argument('positions', type = int, nargs = '+',
                        help = 'Positions')
    parser.add_argument('--delta_before', type = int, default= 6,
                        help = 'Number of frames before collisions')
    parser.add_argument('--delta_after', type = int, default= 9,
                        help = 'Number of frames after collisions')
    args = parser.parse_args()

    out_path = CONFIG['PATHS', 'scenes']

    f = lambda x: compute_timings(x, args.delta_before,
                                  args.delta_after)
    timings = []
    for pos in args.positions:
        pos_suffix = '{0:d}_*/{1:d}_*.json'.format(args.on_ramp, pos)
        scene_paths = glob.glob(os.path.join(out_path, pos_suffix))
        t = list(map(f, scene_paths))
        timings += list(zip(scene_paths, t))

    timing_path = '{0:d}_timings.json'.format(args.on_ramp)
    timing_path = os.path.join(out_path, timing_path)
    with open(timing_path, 'w') as f:
        json.dump(timings, f, sort_keys = True, indent = 2,
                  cls = encoders.NpEncoder)



if __name__ == '__main__':
   main()
