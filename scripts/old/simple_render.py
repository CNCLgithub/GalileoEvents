#!/bin/python3
""" Generates renderings of towers for academic figures.

For a given tower, two sets of images will be generated:

1) With textures and background
2) With wireframe and no background
"""

import os
import sys
import glob
import json
import pprint
import argparse
import shlex
import subprocess
import numpy as np

from utils import config
CONFIG = sys.modules['utils.config'].Config()

#from blockworld.simulation import tower_scene
#from blockworld.simulation.generator import Generator

from blockworld import towers
from blockworld.utils import json_encoders
from experiment.hypothesis.block_hypothesis import simulate


dir_path = os.path.dirname(os.path.realpath(__file__))
render_path = os.path.join(dir_path, 'render.py')

mat_path = os.path.join(dir_path, 'materials.blend')
cmd = '/blender/blender -noaudio --background -P {0!s}'

# Originally imported from `stimuli.analyze_ke`.
# However issues with dask pickle forces me to put this here.

def render(scene_str, traces, out, mode="none", theta=0, frames=[0, 239]):
    if not os.path.isdir(out):
        os.mkdir(out)
    t_path = os.path.join(out, 'trace.json')
    with open(t_path, 'w') as temp:
        json.dump(traces, temp, cls = json_encoders.TowerEncoder)

    fstr = [str(f) for f in frames]

    _cmd = cmd.format(render_path)
    _cmd = shlex.split(_cmd)
    _cmd += [
        '--',
        '--materials',
        mat_path,
        '--out',
        out,
        '--save_world',
        '--frames'
        ] + \
    fstr + \
    ['--scene',
        scene_str,
        '--trace',
        t_path,
        '--resolution',
        '240','240',
        # '1920', '1080',
        '--render_mode',
        # 'motion',
        #'none',
        mode,
        '--theta',
        '{0:f}'.format(theta),
    ]
    subprocess.run(_cmd)
    # subprocess.run(_cmd, check = True)


def get_traces(tower_path):
    """Writes a given trial info to disk

    Arguments:
        tower_path (str): Path to tower
    """

    with open(tower_path, 'r') as f:
        data = json.load(f)
    tower = towers.simple_tower.load(data['struct'])
    trace = data['trace']
    return tower, trace


def simulate_tower(tower_json, output_path, mode="none"):
    """
    Helper function that processes a tower.
    """

    tower, trace = get_traces(tower_json)

    scene_str = json.dumps(tower.serialize())
    render(scene_str, [trace], output_path, mode)



def main():
    parser = argparse.ArgumentParser(
        description = 'Renders the towers in a given directory')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')
    parser.add_argument('--mode', type=str, default='none',
                       choices=['none', 'motion', 'frozen', 'default'],
                       help="Way to render")

    args = parser.parse_args()

    #src = os.path.join(CONFIG['data'], args.src)
    #out = os.path.join(CONFIG['data'], '{0!s}_rendered'.format(args.src))
    # src = args.src
    if os.path.isfile(args.src):
        src = [args.src]
        out = '{0!s}_rendered'.format(os.path.dirname(args.src))
    else:
        src = glob.glob(os.path.join(args.src, '*.json'))# os.path.listdir(args.src)
        out = '{0!s}_rendered'.format(args.src)

    if not os.path.isdir(out):
        os.mkdir(out)

    #for tower_j in glob.glob(os.path.join(src, '*.json')):
    for tower_j in src:
        tower_name = os.path.splitext(os.path.basename(tower_j))[0]
        tower_base = os.path.join(out, tower_name)
        if not os.path.isdir(tower_base):
            os.mkdir(tower_base)
        simulate_tower(tower_j, tower_base, args.mode)

if __name__ == '__main__':
    main()
