import os
import json
import shlex
import argparse
import subprocess
from collections.abc import Iterable

dir_path = os.path.dirname(os.path.realpath(__file__))
render_path = os.path.join(dir_path, 'render.py')

mat_path = os.path.join(dir_path, 'ramp_scene.blend')
cmd = '/blender/blender -noaudio --background -P {0!s}'

def make_args(args_d):
    cmd = ['--', '--save_world']
    for k,v in args_d.items():
        if isinstance(v, str) or not isinstance(v, Iterable):
            cmd += ['--{0!s}'.format(k), str(v)]
        else :
            cmd += ['--{0!s}'.format(k),
                    *[str(e) for e in v]]
    return cmd

def render(**kwargs):
    """ Subprocess call to blender

    Arguments:
        scene_str (str): The serialized tower scene
        traces (dict): A collection of positions and orientations for each
                       block across time.
        theta (float): The camera angle in radians
        out (str): The directory to save renders
    """
    out = ''
    if 'out' in kwargs:
        out += kwargs['out']
    if not os.path.isdir(out):
        os.mkdir(out)
    t_path = os.path.join(out, 'trace.json')
    with open(t_path, 'w') as temp:
        json.dump(kwargs.pop('trace'), temp)

    _cmd = cmd.format(render_path)
    _cmd = shlex.split(_cmd)
    _cmd += make_args(kwargs)
    _cmd += ['--trace', t_path]
    p = subprocess.run(_cmd)