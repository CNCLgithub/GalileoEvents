""" Implementation of Dataset of PF trials.



"""
import io
import os
import json
import time
import argparse
from copy import deepcopy
from pprint import pprint

import h5py
import numpy as np
from h5data import dataset

from blockworld.towers import simple_tower

def get_json(raw):
    byte_array = bytearray(raw)
    s = byte_array.decode("ascii")
    return json.loads(s)

def get_trace(data, raw = False):
    # """Helper that loads numpy byte files"""
    if raw:
        data = get_json(raw)
    return {k : np.array(data[k]) for k in data}

def get_tower(data, raw = False):
    """ Loads raw tower information stored as json"""
    if raw:
        data = get_json(raw)
    return simple_tower.load(data)

def get_block_id(raw):
    return get_json(raw)['block']

class ParticleDataset(dataset.HDF5Dataset):

    """ Implementation of ... for particle filter.

    """

    def __init__(self, source):
        self.source = source

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, path):
        if not os.path.isfile(path):
            raise ValueError('Source does not exist')

        with h5py.File(path, 'r') as f:
            if not 'info' in f['/']:
                raise ValueError('This dataset has no info')
            raw_info = f['/info'].value
        info = get_json(raw_info)
        # Number of scenes * 2
        self.size = info['trials'] * 2
        self._source = path

    @property
    def root(self):
        return '/trials'

    def __len__(self):
        return self.size

    def trials(self, index, handle):
        scene_idx = np.floor(index / 2).astype(int)
        parts = {}
        path = '{0:d}'.format(scene_idx)
        tower = 'org-tower' if index % 2 == 0 else 'mut-tower'
        parts['raw'] = os.path.join(path, tower)
        parts['block'] = os.path.join(path, 'block')
        return parts

    @property
    def trial_funcs(self):
        d = {
            # part of input
            'block' : get_json,
            # target
            'raw' : get_json,
        }
        return d

    def process_trial(self, parts):
        """Returns a tuple containing (observations, tower)"""
        raw = parts['raw']
        trace = get_trace(raw['trace'])
        tower = get_tower(raw['struct'])
        block = parts['block']
        return ((trace, block), tower)


def main():

    parser = argparse.ArgumentParser(
        description = 'Unit test for `ParticleDataset`',
        )
    parser.add_argument('dataset', type = str, help = 'Path to dataset')
    args = parser.parse_args()

    dataset = ParticleDataset(args.dataset)
    print('Dataset has size of {}'.format(len(dataset)))
    init_time = time.time()
    # for t in range(len(dataset)):
    #     dataset[t]
    dataset[:]
    print('All trials accessed in {0:8.5f}s'.format(time.time() - init_time))

if __name__ == '__main__':
    main()
