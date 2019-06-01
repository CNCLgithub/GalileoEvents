"""
Wrapper of forward model for all methods.

Used in generation, rendering, and inference.
"""
import numpy as np
from . import ramp_scene

def simulate(serialized_scene, frames):
    """ Runs pybullet on on given tower
    Frames per second are fixed at 60

    :param serialized_tower: Tower scene to bake
    :param frames: Number of frames to report
    """
    sim = ramp_scene.RampPhysics(serialized_scene)
    blocks = ['ramp', 'table']
    trace = sim.get_trace(frames, blocks, fps = 60, time_step = 240)
    return trace

def simulate_mc(serialized_scene, frames, p):
    """ Runs pybullet on on given tower
    Frames per second are fixed at 60

    :param serialized_tower: Tower scene to bake
    :param frames: Number of frames to report
    """
    sim = ramp_scene.RampPhysics(serialized_scene)
    blocks = ['ramp', 'table']
    trace = sim.get_trace(frames, blocks, fps = 60, time_step = 240)
    return trace
