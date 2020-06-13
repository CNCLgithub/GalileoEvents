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

def simulate(data):
    client = simulation.init_client() # start a server
    sim = simulation.init_sim(Ball3Sim, data, client) # load ramp into client
    trace = simulation.run_full_trace(sim, T = 3.0) # run simulation
    simulation.clear_sim(sim)
    return trace

# todo somehow integrate `vels`
def make_scene(base, objects, positions, vels):
    scene = deepcopy(base)
    for name, (obj, pos, vel) in enumerate(zip(objects, positions, vels)):
        obj = deepcopy(obj)
        scene.add_object(str(name), obj, place=pos, init_vel = vel)
    return scene

def make_ball(appearance, dims, density):
     return shapes.Ball(appearance, dims, {'density': density, 'lateralFriction': 0.3})

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
    base = Ball3World(args.table, args.ramp,
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

    out_path_orig = '/scenes/3ball_change/orig'
    os.path.isdir(out_path_orig) or os.mkdir(out_path_orig)
    out_path_intr = '/scenes/3ball_change/intr'
    os.path.isdir(out_path_intr) or os.mkdir(out_path_intr)
    
    obj_a = make_ball(appearance, dims, 1)
    obj_b = make_ball(appearance, dims, 1)
    obj_c = make_ball(appearance, dims, 1)


    # TODO make function to create a scene given objects
    vels = np.zeros((3, 2, 3))
    scene = make_scene(base, [obj_a, obj_b, obj_c], position, vels)

    p = os.path.join(out_path, '{0:d}.json'.format(0))
    scene_data = scene.serialize() # data must be serialized into a `Dict`
    first_trace = simulate(scene_data) # get first half of simulation
    pal, rot, col = first_trace # for clarity
    # pal is TxS(pos, lin vel, ang vel)xNx3
    # rot is TxNx4
    # col Tx?x2
    contact = np.nonzero(col)[0][0]
    init_pos = pal[contact, 0] # position @ t = contact
    init_rot = rot[contact]

    init_pos_transformed = []
    for pos in init_pos:
        x_coord = pos[0]/3.5
        if x_coord <= 0:
            x_coord = 1 + abs(x_coord)
        init_pos_transformed.append(x_coord)

    # 2 x 3
    init_vels = pal[contact][1:3].transpose() # angular and linear vels @ t = contact

    init_vels_transformed = []
    for vel in init_vels:
        init_vels_transformed.append(vel.transpose())
    # use make_scene
    obj_b_changed = make_ball(appearance, dims, 2)

    scene2 = make_scene(base, [obj_a, obj_b_changed, obj_c], init_pos_transformed, init_vels_transformed)
    trace2 = simulate(scene2.serialize())
    pal2, rot2, col2 = trace2

    # concatenate (np.concatenate)
    # from sim1[0:contact] + sim2

    full_pal = np.concatenate((pal, pal2))
    full_rot = np.concatenate((rot, rot2))
    full_col = np.concatenate((col, col2))

    # save:
    # both scene_datas
    # concatenates simulations
    # two options here, save a dict with json or 3 arrays with np.save

    np.save(file=os.path.join(out_path_orig, "orig_pal.npy"), arr=pal)
    np.save(file=os.path.join(out_path_orig, "orig_rot.npy"), arr=rot)
    np.save(file=os.path.join(out_path_orig, "orig_col.npy"), arr=col)
    
    np.save(file=os.path.join(out_path_intr, "intr_pal.npy"), arr=full_pal)
    np.save(file=os.path.join(out_path_intr, "intr_rot.npy"), arr=full_rot)
    np.save(file=os.path.join(out_path_intr, "intr_col.npy"), arr=full_col)


    # write out metadata
    with open(os.path.join(out_path, 'info'), 'w') as f:
        json.dump({'trials' : 1}, f)

if __name__ == '__main__':
    main()
