#!/usr/bin/env python
""" Evaluates a batch of blocks for stimuli generation.
"""
from galileo_ramp import Exp1Dataset
from physics.world import physics


def profile_scene(scene):
    state = None
    client = physics.init_client()
    obj_ids = physics.init_world(scene, client)
    for _ in range(120):
        physics.update_world(client, obj_ids, scene)
        state = physics.run_mc_trace(client, obj_ids,
                                     state = state)

    physics.clear_trace(client)

def get_scene(dataset):
    scene,state, _ = dataset[0]
    return scene


def main():
    dataset = Exp1Dataset("/databases/exp1.hdf5")
    scene,state, _ = dataset[0]
    mc_state = profile_scene(scene)

if __name__ == '__main__':
   main()
