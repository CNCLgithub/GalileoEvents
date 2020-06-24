#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
from copy import deepcopy
from itertools import chain

from rbw import shapes, worlds, simulation
from rbw.utils.encoders import NpEncoder
from galileo_ramp import BilliardWorld, BilliardSim

surface_phys = {'density' : 0.0,
                'friction': 0.3,
                'restitution' : 0.2}

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
    sim = simulation.init_sim(BilliardSim, data, client) # load ramp into client
    trace = simulation.run_full_trace(sim, T = 3.0) # run simulation
    simulation.clear_sim(sim)
    return trace

def make_scene2(base, objects, positions, vels):
    scene = deepcopy(base)
    for name, (obj, pos, vel) in enumerate(zip(objects, positions, vels)):
        obj = deepcopy(obj)
        scene.add_object(str(name), obj, x=pos[0], y=pos[1], vel = vel)
    return scene

def make_scene(base, objects, positions):
    scene = deepcopy(base)
    for name, (obj, pos) in enumerate(zip(objects, positions)):
        obj = deepcopy(obj)
        if name == 0:
            scene.add_object(str(name), obj, x=pos[0], y=pos[1], vel=[[0, 0, 0], [10, 0, 0]])
        else:
            scene.add_object(str(name), obj, x=pos[0], y=pos[1], vel=[[0, 0, 0], [0, 0, 0]])
    return scene

def make_ball(appearance, dims, density):
     return shapes.Ball(appearance, dims, {'density': density, 'lateralFriction': 0.3,
                                           'restitution' : 0.9})

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates an HDF5 for the Exp 1 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--table', type = int, nargs = 2, default = (4.0, 1.8),
                        help = 'XY dimensions of table.')
    parser.add_argument('--ramp', type = int, nargs = 2, default = (4.0, 1.8),
                        help = 'XY dimensions of ramp.')
    parser.add_argument('--ramp_angle', type = float, default = 35,
                        help = 'ramp angle in degrees')
    args = parser.parse_args()


    # table and table object (`B`) is held constant
    base = BilliardWorld(args.table,
                      table_phys = surface_phys)
    # table_obj = canonical_object("Brick", shapes.Block, obj_dims)
    # base.add_object("B", table_obj, 0.35)

    # different density multiples to change B
    density_changes = [2, 3, 4, 5]
    
    # different starting positions [[[Ball1 x, Ball1 y], [Ball2 x, Ball2 y], [Ball3 x, Ball3 y]]] => 1 trial
    positions = [[[-0.4, 0.0], [0.0, 0.0], [1.0, 0.0]]]
    appearance = "Wood"
    dims = [0.3, 0.3, 0.3]

    out_path = '/scenes/billiard'
    os.path.isdir(out_path) or os.mkdir(out_path)

    obj_a = make_ball(appearance, dims, 1)
    obj_b = make_ball(appearance, dims, 1)
    obj_c = make_ball(appearance, dims, 1)

    i = 0
    # create different trials that are combinations of possible density multiples and starting positions
    for density in density_changes:
        for pos in positions:
            
            # generate scene with initial postions
            scene = make_scene(base, [obj_a, obj_b, obj_c], pos)

            trial_path = os.path.join(out_path, str(i))
            os.path.isdir(trial_path) or os.mkdir(trial_path)
            
            scene_data = scene.serialize() # data must be serialized into a `Dict`

            p = os.path.join(trial_path, 'scene.json')
            with open(p, 'w') as f:
                json.dump(scene_data, f, cls = NpEncoder, indent = 2)

            first_trace = simulate(scene_data) # get first half of simulation
            pal, rot, col = first_trace # for clarity
            # pal is TxS(pos, lin vel, ang vel)xNx3
            # rot is TxNx4
            # col Tx?x2
            contact = np.nonzero(col)[0][0]
            init_pos = pal[contact, 0] # position @ t = contact


            # 2 x 3
            init_vels = np.swapaxes(pal[contact][1:3], 0, 1) # angular and linear vels @ t = contact

            diff = {'objects' : {'1' : {'physics' : {'density' : density}}}}
            with open(os.path.join(trial_path, 'diff.json'), 'w') as f:
                json.dump(diff, f)

            # use make_scene
            obj_b_changed = make_ball(appearance, dims, density)

            scene2 = make_scene2(base, [obj_a, obj_b_changed, obj_c],
                                init_pos, init_vels)

            trace2 = simulate(scene2.serialize())
            pal2, rot2, col2 = trace2

            # concatenate (np.concatenate)
            # from sim1[0:contact] + sim2

            full_pal = np.concatenate((pal[:contact], pal2))
            full_rot = np.concatenate((rot[:contact], rot2))
            full_col = np.concatenate((col[:contact], col2))

            # save:
            # both scene_datas
            # concatenates simulations
            # two options here, save a dict with json or 3 arrays with np.save

            np.save(file=os.path.join(trial_path, "orig_pal.npy"), arr=pal)
            np.save(file=os.path.join(trial_path, "orig_rot.npy"), arr=rot)
            np.save(file=os.path.join(trial_path, "orig_col.npy"), arr=col)
            
            np.save(file=os.path.join(trial_path, "intr_pal.npy"), arr=full_pal)
            np.save(file=os.path.join(trial_path, "intr_rot.npy"), arr=full_rot)
            np.save(file=os.path.join(trial_path, "intr_col.npy"), arr=full_col)

            i+=1


    # write out metadata
    with open(os.path.join(out_path, 'info'), 'w') as f:
        json.dump({'trials' : 4}, f)

if __name__ == '__main__':
    main()
