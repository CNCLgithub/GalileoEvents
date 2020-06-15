import os
import numpy as np

from .exp1_dataset import Exp1Dataset, get_array, get_json

class Ball3Dataset(Exp1Dataset):

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

    def trials(self, index, handle):
        scene_path = '/{0!s}'.format(index)
        parts = dict(
            orig_scene = os.path.join(scene_path, 'scene.json'),
            diff = os.path.join(scene_path, 'diff.json'),
            orig_pal = os.path.join(scene_path, 'orig_pal.npy'),
            orig_rot = os.path.join(scene_path, 'orig_rot.npy'),
            orig_col = os.path.join(scene_path, 'orig_col.npy'),
            intr_pal = os.path.join(scene_path, 'intr_pal.npy'),
            intr_rot = os.path.join(scene_path, 'intr_rot.npy'),
            intr_col = os.path.join(scene_path, 'intr_col.npy'),
            )
        return parts

    @property
    def trial_funcs(self):
        d = dict(orig_scene = get_json,
                 diff = get_json,
                 orig_pal = get_array,
                 orig_rot = get_array,
                 orig_col = get_array,
                 intr_pal = get_array,
                 intr_rot = get_array,
                 intr_col = get_array)
        return d

    def process_trial(self, parts):
        orig_scene = parts['orig_scene']
        diff = parts['diff']
        orig_trace = package_trace(parts['orig_pal'], parts['orig_rot'],
                                   parts['orig_col'])
        intr_trace = package_trace(parts['intr_pal'], parts['intr_rot'],
                                   parts['intr_col'])
        return (orig_scene, diff, orig_trace, intr_trace)

def package_trace(pal, rot, cols):
    trace = dict(
        pos = pal[:, 0],
        orn = rot,
        avl = pal[:, 1],
        lvl = pal[:, 2],
        col = cols
    )
    return trace
