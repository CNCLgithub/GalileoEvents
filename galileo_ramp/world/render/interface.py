import os
import json
import shlex
import argparse
import subprocess
from collections.abc import Iterable
from galileo_ramp.world.simulation.forward_model import TraceEncoder

dir_path = os.path.dirname(__file__)
RENDERFILE = os.path.join(dir_path, 'render.py')

BLENDFILE = os.path.join(dir_path, 'new_scene.blend')
# takes the blend file and the bpy script
cmd = 'xvfb-run -a /blender/blender -noaudio --background {0!s} -P {1!s} -t {2:d}'

def make_args(args_d):
    cmd = ['--', '--save_world']
    for k,v in args_d.items():
        if v is None:
            cmd += ['--{0!s}'.format(k),]
        elif isinstance(v, str) or not isinstance(v, Iterable):
            cmd += ['--{0!s}'.format(k), str(v)]
        else :
            cmd += ['--{0!s}'.format(k),
                    *[str(e) for e in v]]
    return cmd

def render(**kwargs):
    """ Subprocess call to blender
    """
    out = ''
    if 'out' in kwargs:
        out += kwargs['out']
    if not os.path.isdir(out):
        os.mkdir(out)
    t_path = os.path.join(out, 'trace.json')
    with open(t_path, 'w') as temp:
        json.dump(kwargs.pop('trace'), temp,
                  cls = TraceEncoder)

    if 'materials' in kwargs:
        blend_file = kwargs.pop('materials')
    else:
        blend_file = BLENDFILE

    if 'render' in kwargs:
        render_path = kwargs.pop('render')
    else:
        render_path = RENDERFILE

    if 'threads' in kwargs:
        threads = kwargs.pop('threads')
    else:
        threads = len(os.sched_getaffinity(0))

    _cmd = cmd.format(blend_file, render_path, threads)
    _cmd = shlex.split(_cmd)
    _cmd += make_args(kwargs)
    _cmd += ['--trace', t_path]
    print('Running blender')
    p = subprocess.run(_cmd)
