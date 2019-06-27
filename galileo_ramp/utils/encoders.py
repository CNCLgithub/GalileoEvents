""" Defines JSON encoders used throughout the project
"""

import json
import numpy as np

class NpEncoder(json.JSONEncoder):

    """ Encodes numpy elements for json """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)
