#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
from copy import deepcopy
from itertools import chain

from rbw import shapes, worlds, simulation
from rbw.utils.encoders import NpEncoder

surface_phys = {'density' : 0.0,
                'friction': 0.3}
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
    return np.linspace(1.5, 1.8, num = n)

def make_pair(scene, material, shp, density, pos):
    dims = sample_dimensions(obj_dims)
    congruent = canonical_object(material, shp, dims)
    incongruent = shapes.shape.change_prop(congruent, 'density', density)
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
    base = worlds.RampWorld(args.table, args.ramp,
                     ramp_angle = args.ramp_angle * (np.pi/180.),
                     ramp_phys = surface_phys,
                     table_phys = surface_phys)
    table_obj = canonical_object("Brick", shapes.Block, obj_dims)
    base.add_object("B", table_obj, 0.35)

    # materials have the same proportions of heavy/light perturbations
    densities = np.hstack((low_densities(5),
                           high_densities(5)))
    positions = interpolate_positions(5)
    positions = np.repeat(positions, 2)

    # generate the 60 pairs of ramp objects
    pairs = []
    for m in ['Iron', 'Brick', 'Wood']:
        for shp in [shapes.Block, shapes.Puck]:
            for dp in zip(densities, positions):
                pairs.append(make_pair(base, m, shp, *dp))

    # generate the 90 control trials (not paired/matched)
    controls = []
    positions = interpolate_positions(5)
    positions = np.repeat(positions, 3)
    for m in ['Iron', 'Brick', 'Wood']:
        for shp in [shapes.Block, shapes.Puck]:
        # 15 vs 30 since there are 2 (block+puck) per loop
            for p in positions:
                controls.append(make_control(base, m, shp, p))

    # for each scene (pair or control) randomly pick an initial
    # positions for object `A`
    scenes = list(chain.from_iterable(pairs))
    print(len(scenes))
    scenes += controls
    print(len(scenes))
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
            json.dump(data, f, indent = 2, cls = NpEncoder)

    # write out metadata
    with open(out_path + 'info', 'w') as f:
        json.dump({'trials' : len(scenes)}, f)

if __name__ == '__main__':
    main()
