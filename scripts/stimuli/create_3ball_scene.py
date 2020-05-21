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
    # table_obj = canonical_object("Brick", shapes.Block, obj_dims)
    # base.add_object("B", table_obj, 0.35)

    # materials have the same proportions of heavy/light perturbations
    density_ratios = ...

    positions = ... # [[1,2,3]]

    out_path = '/scenes/3ball'
    os.path.isdir(out_path) or os.mkdir(out_path)

    # makes 20 scenes ...
    for i in range(20):
        scene = ...
        p = os.path.join(out_path, '{0:d}.json'.format(i))
        data = {'scene' : scene.serialize()}
        with open(p, 'w') as f:
            json.dump(data, f, indent = 2, cls = NpEncoder)
        pass


    # write out metadata
    with open(out_path + 'info', 'w') as f:
        json.dump({'trials' : 20}, f)

if __name__ == '__main__':
    main()
