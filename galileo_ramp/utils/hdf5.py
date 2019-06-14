import os
import io
import h5py
import pprint
from contextlib import contextmanager

import numpy as np

class HDF5:

    def __init__(self, path, overwrite=False):

        self.overwrite = overwrite
        self.out = path


    @property
    @contextmanager
    def out(self):
        with h5py.File(self._out_path, 'a') as f:
            yield f

    @out.setter
    def out(self, o):
        if not isinstance(o, io.BytesIO):
            if os.path.isfile(o) and self.overwrite:
                msg = 'Found previous result file. Deleting {}'.format(o)
                print(msg)
                os.remove(o)
        self._out_path = o


    def write(self, d):
        with self.out as f:
            to_hdf5(f, d)

def to_dic(data):
    """
    Helper function that does the inverse of `from_dic`
    """
    d = {}
    for grp in data:
        dset = data[grp]
        if isinstance(dset, h5py.Group):
            d.update( {grp : to_dic(dset)} )
        else:
            v = dset.value
            d.update( {grp : v })

    return d

def to_hdf5(h5, dic):
    """
    Helper function that type-casts certain values from a `dict` to hdf5
    """
    for key, item in dic.items():

        if not isinstance(key, str):
            try:
                key = '{}'.format(key)
            except:
                print(key)
                raise ValueError('Key not valid')

        if key in h5.keys():
            return
        else:
            if isinstance(item, (np.ndarray, np.intc, np.float64, str, bytes,
                int, list, float, np.int64)):
                h5[key] = np.array(item).squeeze()
            elif item is None:
                h5[key] = "None"
            elif isinstance(item, dict):
                g = h5.create_group(key)
                to_hdf5(g, item)
            else:
                print(dic)
                raise ValueError('Cannot save {0!s} type from key {1!s}'.format(type(item), key))
