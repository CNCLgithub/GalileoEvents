#!/bin/python3
""" Generates renderings of towers for academic figures.

For a given tower, two sets of images will be generated:

1) With textures and background
2) With wireframe and no background
"""

import os
import glob
import json
import pprint
import argparse
import numpy as np

from config import Config
from blockworld.simulation import block_scene
from blockworld.simulation.generator import Generator

CONFIG = Config()

def simulate_tower(tower, path):
    """
    Helper function that processes a tower.
    """

    with open(tower, 'r') as f:
        tower_json = json.load(f)

    scene = block_scene.BlockScene(tower_json, wire_frame = False, frames = 120)
    blend_path = os.path.join(path, 'scene.blend')
    scene.bake_physics()
    # frozen_path = os.path.join(path, 'frozen')
    # scene.render_circle(frozen_path, freeze = True, dur = 2,
    #                     resolution = (256, 256))
    # motion_path = os.path.join(path, 'motion')
    # scene.render(motion_path, np.arange(120), resolution = (256, 256))
    scene.save(blend_path)



def main():
    parser = argparse.ArgumentParser(
        description = 'Renders the towers in a given directory')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')

    args = parser.parse_args()

    src = os.path.join(CONFIG['data'], args.src)
    out = os.path.join(CONFIG['data'], '{0!s}_rendered'.format(args.src))
    # src = args.src
    # out = '{0!s}_rendered'.format(args.src)

    if not os.path.isdir(out):
        os.mkdir(out)

    for tower_j in glob.glob(os.path.join(src, '*.json')):
        tower_name = os.path.splitext(os.path.basename(tower_j))[0]
        tower_base = os.path.join(out, tower_name)
        if not os.path.isdir(tower_base):
            os.mkdir(tower_base)
        simulate_tower(tower_j, tower_base)

if __name__ == '__main__':
    main()
