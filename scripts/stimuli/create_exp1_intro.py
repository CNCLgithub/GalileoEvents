#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
from copy import deepcopy
from itertools import chain

from physics.world.scene import shape
from physics.world.scene.puck import Puck
from physics.world.scene.block import Block
from physics.world.scene.ramp import RampScene
from physics.world.simulation import physics
from physics.utils import encoders

surface_phys = {'density' : 0.0,
                'friction': 0.34}
obj_dims = np.array([3.0, 3.0, 1.5]) / 10.0
density_map = {"Wood" : 1.0,
               "Brick" : 2.0,
               "Iron" : 8.0}
friction_map = {"Wood" : 0.263,
                "Brick" : 0.323,
                "Iron" : 0.215}

def canonical_object(material, shape, dims):
    return shape(material, dims, {'density' : density_map[material],
                                  'lateralFriction' : friction_map[material],
                                  'restitution' : 0.9})

def low_densities(n):
    return np.exp(np.linspace(-5.5, -5.0, num = n))

def high_densities(n):
    return np.exp(np.linspace(5.0, 5.5, num = n))

def sample_dimensions(base):
    bound = np.log(1.6)
    samples = np.exp(np.random.uniform(-1*bound,bound, size = 3))
    return base * samples

def interpolate_positions(n):
    return np.linspace(1.3, 1.7, num = n)

def make_pair(scene, material, shp, density, pos):
    dims = sample_dimensions(obj_dims)
    congruent = canonical_object(material, shp, dims)
    incongruent = shape.change_prop(congruent, 'density', density)
    con = deepcopy(scene)
    con.add_object('A', congruent, pos)
    incon = deepcopy(scene)
    incon.add_object('A', incongruent, pos)
    return [con, incon]

def make_control(scene, material, shape, pos):
    dims = sample_dimensions(obj_dims)
    obj = canonical_object(material, shape, dims)
    s = deepcopy(scene)
    s.add_object('A', obj, pos)
    return s

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates an HDF5 for the Exp 1 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--table', type = int, nargs = 2, default = (3.5, 1.8),
                        help = 'XY dimensions of table.')
    parser.add_argument('--ramp', type = int, nargs = 2, default = (3.5, 1.8),
                        help = 'XY dimensions of ramp.')
    parser.add_argument('--ramp_angle', type = float, default = 35,
                        help = 'ramp angle in degrees')
    args = parser.parse_args()


    # table and table object (`B`) is held constant
    base = RampScene(args.table, args.ramp,
                     ramp_angle = args.ramp_angle * (np.pi/180.),
                     ramp_phys = surface_phys,
                     table_phys = surface_phys)
    table_obj = canonical_object("Brick", Block, obj_dims)
    base.add_object("B", table_obj, 0.4)

    # materials have the same proportions of heavy/light perturbations
    densities = np.hstack((low_densities(5),
                           high_densities(5)))


    # canonical scene iron block
    scene_1 = make_control(base, 'Iron', Block, 1.5)

    # wood heavier than brick
    scene_2 = make_pair(base, 'Wood', Block, 20.0, 1.4)[1]

    # another position and material
    scene_3 = make_control(base, 'Brick', Puck, 1.7)

    # practice trials
    mats = ['Brick', 'Iron', 'Wood', 'Wood']
    shapes = [Puck, Block, Puck, Block]
    positions = interpolate_positions(4)

    practice_scenes = [make_control(base, m, shp, p)
                       for (m,shp,p) in zip(mats,shapes,positions)]

    # save trials to json
    out_path = '/scenes/exp1_intro/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    scenes = [scene_1, scene_2, scene_3, *practice_scenes]
    for i,s in enumerate(scenes):
        p = os.path.join(out_path, '{0:d}.json'.format(i))
        data = {'scene' : s.serialize()}
        with open(p, 'w') as f:
            json.dump(data, f, indent = 2, cls = encoders.NpEncoder)

    # write out metadata
    with open(out_path + 'info', 'w') as f:
        json.dump({'trials' : len(scenes)}, f)

if __name__ == '__main__':
    main()
