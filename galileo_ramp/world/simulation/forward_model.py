"""
Wrapper of forward model for all methods.

Used in generation, rendering, and inference.
"""
import json
import numpy as np
from .ramp_physics import RampPhysics

def simulate(serialized_scene, frames, debug = False):
    """ Runs pybullet on on given tower
    Frames per second are fixed at 60

    :param serialized_tower: Tower scene to bake
    :param frames: Number of frames to report
    """
    sim = RampPhysics(serialized_scene, debug = debug)
    # ensures that the objects are reported in order
    objs = list(map(str, range(len(serialized_scene['objects']))))
    trace = sim.get_trace(frames, objs, fps = 60)
    return trace

def simulate_mc(serialized_scene, frames, p):
    """ Runs pybullet on on given tower
    Frames per second are fixed at 60

    :param serialized_tower: Tower scene to bake
    :param frames: Number of frames to report
    """
    sim = RampPhysics(serialized_scene)
    blocks = ['ramp', 'table']
    trace = sim.get_trace(frames, blocks, fps = 60)
    return trace


class TraceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
