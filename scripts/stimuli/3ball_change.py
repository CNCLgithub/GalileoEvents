#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
from copy import deepcopy
from itertools import chain

from rbw import shapes, worlds, simulation
from rbw.utils.encoders import NpEncoder
from galileo_ramp import Ball3World, Ball3Sim

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
    density_ratios = [1, 1, 1]
    position = [1.9, 0.9, 0.2]
    appearance = "Wood"
    dims = [0.3, 0.3, 0.3]

    out_path = '/scenes/3ball_change'
    os.path.isdir(out_path) or os.mkdir(out_path)
    
    obj_a = shapes.Ball(appearance, dims, {'density': density_ratios[0], 'lateralFriction': 0.3})
    obj_b = shapes.Ball(appearance, dims, {'density': density_ratios[1], 'lateralFriction': 0.3})
    obj_c = shapes.Ball(appearance, dims, {'density': density_ratios[2], 'lateralFriction': 0.3})

    scene=deepcopy(base)

    scene.add_object('A', obj_a, position[0])
    scene.add_object('B', obj_b, position[1])
    scene.add_object('C', obj_c, position[2])

    p = os.path.join(out_path, '{0:d}.json'.format(0))
    scene_data = scene.serialize() # data must be serialized into a `Dict`
    print(scene_data)
    client = simulation.init_client(debug = True) # start a server
    sim = simulation.init_sim(simulation.MarbleSim, scene_data, client) # load ramp into client
    print(sim)

    pla, rot, col = simulation.run_full_trace(sim, debug = True) # run simulation
    print(col)
    # # print(trace.shape)
    simulation.clear_sim(sim)







    with open(p, 'w') as f:
        json.dump(scene_data, f, indent = 2, cls = NpEncoder)


    # write out metadata
    with open(os.path.join(out_path, 'info'), 'w') as f:
        json.dump({'trials' : 1}, f)

if __name__ == '__main__':
    main()
