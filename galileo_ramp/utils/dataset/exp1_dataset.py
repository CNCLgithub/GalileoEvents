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
import numpy as np

# Defined for ratios
variables = ["Density", "Mass", "Friction"]

def get_array(bs):
    s = bs.tostring()
    f = io.BytesIO(s)
    v = np.load(f)
return v

class Exp1Dataset(dataset.HDF5Dataset):

    '''
    source: Self-contained hdf5 dataset. Means are expected
                under the root path.
    debug:  Flag to run in debug mode.
    INPUT FORMAT:
    256
    LABEL FORMAT:
    4 + 6 + 6 + 2 + 1 + 2 = 21
    (params, dims, pos, rots, cols, prnts)
    '''

    def __init__(self, source, root = '/', no_check=True, ratio = True,
                 features = variables, n = 210):
        self.ratio = ratio
        self.features = features
        self.source = source
        self.root = root
        self.size = n

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, path):
        if not os.path.isfile(path):
            raise ValueError('Source does not exist')

        with h5py.File(path, 'r') as f:
            if not 'info' in f['/']:
                print('This dataset has no info')
            else:
                raw_info = f['/info'].value
                info = get_json(raw_info)
                # Number of scenes * 2
                self.size = info['trials'] * 2
        self._source = path

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
        d = {	# inputs
            'position' : get_array, # (objs, frames, 3)
            'velocity' : get_array, # (objs, frames, 3)
            'col' : get_array,
        }
        return d

    def process_trial(self, parts):

        pos = np.vstack(parts['position'])
        vel = np.vstack(parts['velocity'])
        col = np.vstack(parts['col']).flatten()

        moves_ramp = BlenderStimuli.tools.frames.moving(vel[0])
        moves_ground = BlenderStimuli.tools.frames.moving(vel[1])
        moves = np.logical_or(moves_ramp, moves_ground)
        ending = np.where(moves)[0][-1]
        contact = col[0]
        n_frames = len(pos[0])

        return (contact, ending, n_frames)

    # def get_param(self, param):

    #     param = super(HumanDataset, self).get_param(param)

    #     target_features = self.features

    #     values = {}
    #     objects = param['Objects']

    #     if self.ratio:
    #         # compute ratio between A and B or just report A
    #         for feature in objects['A']:
    #             a_val = objects['A'][feature]

    #             if feature in target_features:
    #                 b_val = objects['B'][feature]
    #                 values[feature] = a_val / b_val
    #             else:
    #                 if feature  == "Shape":
    #                     a_val = ramp_shape.shapes[a_val]
    #                 elif feature == "Material":
    #                     a_val = ramp_shape.materials[a_val]

    #                 values[feature] = a_val
    #     else:
    #         values = objects
    #     return values

def main():

    parser = argparse.ArgumentParser(
        description = 'Unit test for `ParticleDataset`',
        )
    parser.add_argument('dataset', type = str, help = 'Path to dataset')
    args = parser.parse_args()

    dataset = Exp1Dataset(args.dataset, n = 210)
    print('Dataset has size of {}'.format(len(dataset)))
    init_time = time.time()
    # for t in range(len(dataset)):
    #     dataset[t]
    dataset[:]
    print('All trials accessed in {0:8.5f}s'.format(time.time() - init_time))

if __name__ == '__main__':
    main()
