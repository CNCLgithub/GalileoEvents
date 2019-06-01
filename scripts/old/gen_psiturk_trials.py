""" Creates the set of psiturk trials.
"""

import os
import json
import glob
import argparse
import numpy as np
from pprint import pprint
from itertools import repeat

from utils import config
CONFIG = config.Config()

def main():

    parser = argparse.ArgumentParser(
        description = 'Creates a psiturk trial list from a PF run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('movies', type = str,
                        help = 'Paths to the PF movies')
    parser.add_argument('--conditions', type = int, default = 8,
                        help = 'Number of conditions.')
    args = parser.parse_args()

    # Generates the pattern for movies for a given trial
    str_base = lambda x: 'trial_{0:d}_cond_*.mp4'.format(x)
    gen_cond = lambda p,c: 'trial_{0:d}_cond_{1:d}.mp4'.format(p,c)
    # The pairs of trials that can be put into psiturk
    scenes = []
    for t_ind in np.arange(0, 239, step = 2):
        # Search to see if each part of the pair has the right number
        # of conditions (n = 4)
        con =os.path.join(args.movies, str_base(t_ind))
        con = glob.glob(con)
        incon = os.path.join(args.movies, str_base(t_ind + 1))
        incon = glob.glob(incon)
        if not (len(con) == args.conditions/2 and \
                len(incon) == args.conditions/2):
            msg = 'Missing component for {0:d}, ({1:d}+{2:d}) != {3:d}'
            msg = msg.format(t_ind, len(con), len(incon), args.conditions)
            print(msg)
            continue

        # add the different conditions to the scene list
        a = np.array(list(map(gen_cond, repeat(t_ind),
                         np.arange(args.conditions/2, dtype = int))))
        b = np.array(list(map(gen_cond, repeat(t_ind+1),
                          np.arange(args.conditions/2, dtype = int))))
        conds = np.zeros((a.size + b.size,), dtype = a.dtype)
        conds[0::2] = a
        conds[1::2] = b
        scenes.append(conds)

    print('Found {0:d} trials'.format(len(scenes)))
    trials = np.empty((args.conditions, len(scenes)), dtype = object)

    for c in np.arange(args.conditions):
        for p_ind, scene in enumerate(scenes):
            v = (p_ind + c) % args.conditions
            trials[c,p_ind] = scene[v]

    # out = os.path.join(CONFIG['PATHS', 'root'], 'psiturk', 'static', 'json',
    #                    'condlist.json')
    out = os.path.join(args.movies, 'condlist.json')
    if os.path.isfile(out):
        print('Overwriting condlist at: {}'.format(out))

    with open(out, 'w') as f:
        json.dump(trials.tolist(), f)

if __name__ == '__main__':
    main()
