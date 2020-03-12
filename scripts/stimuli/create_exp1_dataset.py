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
    return np.exp(np.random.uniform(-2.5, -1.5, size = n))

def high_densities(n):
    return np.exp(np.random.uniform(1.5, 2.5, size = n))

def sample_dimensions(base):
    bound = np.log(1.6)
    samples = np.exp(np.random.uniform(-1*bound,bound, size = 3))
    return base * samples

def sample_position():
    return np.random.uniform(0.3, 0.7) + 1

def make_pair(material, density, shp):
    dims = sample_dimensions(obj_dims)
    congruent = canonical_object(material, shp, dims)
    incongruent = shape.change_prop(congruent, 'density', density)
    return (congruent, incongruent)

def make_control(material, shape):
    dims = sample_dimensions(obj_dims)
    return canonical_object(material, shape, dims)

def from_pair(scene, pair):
    ramp_pos = sample_position()
    con = deepcopy(scene)
    con.add_object('A', pair[0], ramp_pos)
    incon = deepcopy(scene)
    incon.add_object('A', pair[0], ramp_pos)
    return [con, incon]

def from_control(scene, obj):
    ramp_pos = sample_position()
    s = deepcopy(scene)
    s.add_object('A', obj, ramp_pos)
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
    densities = np.hstack((low_densities(10),
                           high_densities(10)))

    # generate the 60 pairs of ramp objects
    pairs = []
    for m in ['Iron', 'Brick', 'Wood']:
        for d in densities:
            block_pair = make_pair(m, d, Block)
            pairs.append(block_pair)
            puck_pair = make_pair(m, d, Puck)
            pairs.append(puck_pair)

    controls = []
    # generate the 90 control trials (not paired/matched)
    for m in ['Iron', 'Brick', 'Wood']:
        # 15 vs 30 since there are 2 (block+puck) per loop
        for _ in range(15):
            block_obj = make_control(m, Block)
            controls.append(block_obj)
            puck_obj = make_control(m, Puck)
            controls.append(puck_obj)

    # for each scene (pair or control) randomly pick an initial
    # positions for object `A`
    # scenes = list(reduce(sum, map(lambda p: from_pair(base, p), pairs)))
    scenes = map(lambda p: from_pair(base, p), pairs)
    scenes = list(chain.from_iterable(scenes))
    scenes += list(map(lambda x: from_control(base, x), controls))

    # save trials to json
    out_path = '/scenes/exp1/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for i,s in enumerate(scenes):
        p = os.path.join(out_path, '{0:d}.json'.format(i))
        # trace  = physics.run_full_trace(s.serialize(),
        #                               ['A', 'B'],
        #                               fps = 60,
        #                               time_scale = 1.0,
        #                               debug = True)
        data = {'scene' : s.serialize()}
        with open(p, 'w') as f:
            json.dump(data, f, indent = 2, cls = encoders.NpEncoder)


if __name__ == '__main__':
    main()
