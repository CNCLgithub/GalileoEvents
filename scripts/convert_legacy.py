#!/usr/bin/env python
""" Converts legacy json into current scene semantics

"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob

from galileo_ramp.utils import config
from galileo_ramp.world.scene.puck import Puck
from galileo_ramp.world.scene.block import Block
from galileo_ramp.world.scene.ramp import RampScene
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

# Data extracted manually from:
# git@github.mit.edu:k2smith/HumanGalileo.git -> BlenderStimuli/world/ramp_scene.blend
block_dims = (3, 3, 1.5) # x,y,z
puck_dims = (3, 3, 1.5)
table_dims = (35, 18)
table_friction = 0.340
ramp_dims = (32, 18)
ramp_friction = 0.340
ramp_angle = 15 # degrees



# From https://github.mit.edu/k2smith/HumanGalileo/blob/ddf8e7e79d0e35aa7bbac473a8d32346cee6f0eb/BlenderStimuli/shapes/ramp_shape.py
appearance_map = ["Iron", "Brick", "Wood"]
shapes =  ["Block", "Puck"]
shape_map = [
    # block constructor
    Block,
    # puck contructor
    Puck
]

def convert_scene(old_file, dest):
    with open(old_file, 'r') as f:
        data = json.load(f)

    # First reorganize "scene" data
    objects = data['Objects']
    scene = RampScene(table_dims, ramp_dims,
                      ramp_angle = ramp_angle * (np.pi/180.),
                      table_friction = table_friction,
                      ramp_friction = ramp_friction)

    for obj_name, obj_data in objects.items():

        # Construct the shape
        appearance = appearance_map[obj_data['Material']]
        density = obj_data['Density']
        dims = obj_data['Scaling']
        friction = obj_data['Friction']
        shape = shape_map[obj_data['Shape']](appearance, dims, density, friction)

        # Place object on the table or ramp
        pct = obj_data['SlopePos']
        # Shape `A` is on the ramp in old syntax
        if obj_name == 'A':
            pct += 1 # indicates in the current syntax object is on ramp
        else:
            pct = 1 - pct # table is reversed
        scene.add_object(obj_name, shape, pct)

    result = { 'scene' : scene.serialize() }
    r_str = json.dumps(result, indent = 4, sort_keys = True,
                       cls = forward_model.TraceEncoder)
    filename = os.path.basename(old_file)
    out = os.path.join(dest, filename)
    with open(out, 'w') as f:
        f.write(r_str)


def main():

    parser = argparse.ArgumentParser(
        description = 'Converts legacy json into current scene semantics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str,
                        help = 'Path to legacy jsons')
    parser.add_argument('--destination', '-d', type = str, default = 'legacy',
                        help = 'Path to save converted scenes')
    args = parser.parse_args()

    out_path = os.path.join(CONFIG['PATHS', 'scenes'], args.destination)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    legacy_scenes = glob(os.path.join(args.src, '*.json'))
    for scene in legacy_scenes:
        convert_scene(scene, out_path)

if __name__ == '__main__':
   main()
