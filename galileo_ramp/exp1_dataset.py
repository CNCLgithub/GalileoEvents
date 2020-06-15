import io
import os
import json
import h5py
import numpy as np
from pprint import pprint
from h5data import dataset

from rbw import simulation
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
        client = simulation.init_client()
        sim = simulation.init_sim(simulation.RampSim, scene, client) # load ramp into client
        pla,rot,cols = simulation.run_full_trace(sim, T = 2.0, fps = 60,
                                                 time_scale = 1.0)
        simulation.clear_sim(sim)
        trace = dict(
            pos = pla[:, 0],
            orn = rot,
            avl = pla[:, 1],
            lvl = pla[:, 2],
            col = cols
        )
        contact = np.nonzero(trace['col'])[0][0]
        time_points = np.array([-1, 1, 3, 5]) * self.time_scale
        time_points += contact
        return (scene, trace, time_points)
