import io
import os
import json
import h5py
import time
import argparse
import numpy as np
from pprint import pprint
from h5data import dataset

from physics.world.simulation import physics
import numpy as np

def get_json(raw):
    byte_array = bytearray(raw)
    s = byte_array.decode("ascii")
    return json.loads(s)

def get_array(bs):
    s = bs.tostring()
    f = io.BytesIO(s)
    v = np.load(f)
    return v

def is_moving(vs):
    # Frames of ground movement
    moving = np.greater(np.abs(vs.mean(axis=-1)), 1E-5).flatten()
    return moving

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

    def __init__(self, source, no_check=True, ratio = True,
                 n = 210):
        self.ratio = ratio
        self.source = source
        self.size = n

    @property
    def root(self):
        return '/'

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
                raw_info = f['/info'][()]
                info = get_json(raw_info)
                self.size = info['trials']
        self._source = path

    def __len__(self):
        return self.size

    def trials(self, index, handle):
        scene_idx = int(index)
        scene_json =  '/{0:d}.json'.format(scene_idx)
        parts = {
            'scene' : scene_json
        }
        return parts

    @property
    def trial_funcs(self):
        return {
            'scene' : get_json
        }

    def process_trial(self, parts):


        scene = parts['scene']['scene']
        trace = physics.run_full_trace(scene,
                                       ['A', 'B'],
                                       T = 2,
                                       fps = 60,
                                       time_scale = 1.0,
                                       debug = False)
        pos, rot, ang_vel, vel, col = trace

        moves_ramp = is_moving(vel[:,0])
        moves_ground = is_moving(vel[:,1])
        moves = np.logical_or(moves_ramp, moves_ground)
        ending = np.where(moves)[0][-1]
        contact = np.nonzero(col)[0][0]

        n_frames = len(pos)

        time_points = [contact -1,
                       contact + 12,
                       int((contact + ending)/2),
                       min(ending + 12, n_frames)]
        return (scene, trace, time_points)


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
    print(dataset[0][-1])

if __name__ == '__main__':
    main()
