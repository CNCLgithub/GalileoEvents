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
from datasets import particle_dataset

def get_json(raw):
    byte_array = bytearray(raw)
    s = byte_array.decode("ascii")
    return json.loads(s)


def get_trace(data, raw = False):
    """Helper that loads numpy byte files"""
    if raw:
        data = get_json(data)
    return {k : np.array(data[k]) for k in data}

def get_tower(data, raw = False):
    """ Loads raw tower information stored as json"""
    if raw:
        data = get_json(data)
    return simple_tower.load(data)

def get_block_id(raw):
    return get_json(raw)['block']

class SimpleDataset(particle_dataset.ParticleDataset):

    """ Implementation of ... for dataset analysis.

    """

    components = ['struct', 'trace']
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
        # Number of scenes
        self.size = info['trials']
        self._source = path

    def trials(self, index, handle):
        path = '{0:d}'.format(index)
        parts = {k : os.path.join(path, k) for k in self.components}
        return parts

    @property
    def trial_funcs(self):
        # return {k : get_json for k in self.components}
        d = {
            'struct' : lambda t: get_tower(t, raw = True),
            'trace'  : lambda t: get_trace(t, raw = True)
        }
        return d

    def process_trial(self, parts):
        """Returns a tuple containing (observations, tower)"""
        left = parts['struct']
        right = parts['trace']
        return (left, right)


def main():

    parser = argparse.ArgumentParser(
        description = 'Unit test for `ParticleDataset`',
        )
    parser.add_argument('dataset', type = str, help = 'Path to dataset')
    args = parser.parse_args()

    dataset = SimpleDataset(args.dataset)
    print('Dataset has size of {}'.format(len(dataset)))
    init_time = time.time()
    # for t in range(len(dataset)):
    #     dataset[t]
    dataset[:]
    print('All trials accessed in {0:8.5f}s'.format(time.time() - init_time))

if __name__ == '__main__':
    main()
