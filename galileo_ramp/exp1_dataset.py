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
                 time_scale = 6):
        self.ratio = ratio
        self.source = source
        self.time_scale = time_scale

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
                raise ValueError('This dataset has no info')

            raw_info = f['/info'][()]
            info = get_json(raw_info)
            self.size = info['trials']
        self._source = path

    @property
    def trace_features(self):
        return ['pos', 'orn', 'avl', 'lvl']

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
        client = physics.init_client(direct = True)
        obj_ids = physics.init_world(scene, client)
        state,cols = physics.run_full_trace(client,
                                            obj_ids,
                                            T = 2,
                                            fps = 60,
                                            time_scale = 1.0,
                                            debug = False)
        physics.clear_trace(client)
        trace = {k:state[:,i] for i,k in enumerate(self.trace_features)}
        trace['col'] = cols
        contact = np.nonzero(trace['col'])[0][0]
        time_points = np.array([-1, 1, 3, 5]) * self.time_scale
        time_points += contact
        return (scene, trace, time_points)

def main():

    parser = argparse.ArgumentParser(
        description = 'Unit test for `ParticleDataset`',
        )
    parser.add_argument('dataset', type = str, help = 'Path to dataset')
    args = parser.parse_args()

    dataset = Exp1Dataset(args.dataset)
    print(dataset[28])
    print('Dataset has size of {}'.format(len(dataset)))
    init_time = time.time()
    dataset[:]
    print('All trials accessed in {0:8.5f}s'.format(time.time() - init_time))

if __name__ == '__main__':
    main()
