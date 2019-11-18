#!/usr/bin/env python3

""" Ensures that the each condition is balanced across temporal groups
"""

import os
import json
import argparse
import numpy as np

from galileo_ramp.utils import config, encoders
CONFIG = config.Config()

def parse_file_str(fp):
    """ Parses trial info from file path

    Returns an integer representing the timing group for that trial.
    """
    base = os.path.basename(fp)
    base = base[:-4] # remove ext
    parts = base.split('_')
    # the last digit notes the timing group
    tg = parts[-1]
    return tg


def main():

    parser = argparse.ArgumentParser(
        description = "Ensures that the each condition is balanced across temporal groups",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--condlist', type = str, default = 'condlist.json',
                        help = 'Name of trial list')
    args = parser.parse_args()
    condlist_path = os.path.join(CONFIG['PATHS', 'scenes'], args.condlist)

    with open(condlist_path, 'r') as f:
        condlist = json.load(f)

    for idx, cond in enumerate(condlist):
        # Each condition is a list of tuples (file path, [colors])
        paths, _ = zip(*cond)
        tgs = list(map(parse_file_str, paths))
        # Obtain unique counts
        groups, counts = np.unique(tgs, return_counts = True)
        balance = dict(zip(groups, counts))
        r_str = json.dumps(balance, indent = 4, sort_keys = True,
                           cls = encoders.NpEncoder)
        msg = 'Balance for condition {0:d}:\n\t{1!s}'.format(idx,
                                                             r_str)
        print(msg)


if __name__ == '__main__':
    main()
