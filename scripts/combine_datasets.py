#!/usr/bin/env python3
""" Combines a set of datasets."""

import os
import csv
import glob
import json
import h5py
import shutil
import argparse
import datetime
import numpy as np
from pprint import pprint

from utils import config

CONFIG = config.Config()



def retrieve_trials(sources):
    """ Collects lists of valid trials.

    Arguments:
        source (str): Path to csv listing trials
    """
    d = []
    for source in sources:
        name = source.replace('_valid.csv', '.hdf5')
        trials = []
        with open(source, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                try:
                    trials.append(int(int(row[2])/2))
                except:
                    continue
            d.append((name, trials))
    return d


def copy_trials(path, trials, n, f):
    p_str = '/trials/{0:d}'
    with h5py.File(path, 'r') as g:
        if n == 0:
            dest = f.create_group('/trials')
        else:
            dest = f['/trials']
        for i,t in enumerate(trials):
            source = p_str.format(t)
            g.copy(source, dest, str(n + i))

def main():
    parser = argparse.ArgumentParser(
        description = 'Creates tower datasets')

    parser.add_argument('sources', type = str, nargs = '+',
                        help = 'Paths to tower jsons')
    parser.add_argument('--out', type = str,
                        help = 'Path to save dataset.')
    args = parser.parse_args()

    if not args.out is None:
        out = args.out
    else:
        suffix = datetime.datetime.now().strftime("%m_%d_%y_%H%M%S")
        out = "_".join(["combinded", suffix]) + '.hdf5'
        out = os.path.join(CONFIG['PATHS', 'databases'], out)

    sources = [os.path.join(CONFIG['PATHS', 'databases'], p)
               for p in args.sources]
    trials = retrieve_trials(sources)
    print(trials)
    print('Saving to {}'.format(out))
    n_trials = 0
    with h5py.File(out, 'w') as f:
        for data_path, ids in trials:
            copy_trials(data_path, ids, n_trials, f)
            n_trials += len(ids)
        info = {
            'trials' : n_trials,
            'sources' : args.sources,
            'time' : datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        }
        info_s = bytearray(json.dumps(info), encoding = 'ascii')
        f.create_dataset('/info', data = np.void(info_s))


if __name__ == '__main__':
    main()
