#!/usr/bin/env python
""" Evaluates a batch of blocks for stimuli generation.
"""
from galileo_ramp import Exp1Dataset
from physics.world import physics


def profile_scene(scene):
    state = None
    client, obj_ids = physics.initialize_trace(scene)
    for _ in range(120):
        physics.update_world(client, obj_ids, scene)
        state = physics.run_mc_trace(client, obj_ids,
                                     state = state)
    physics.clear_trace(client)

def get_scene(dataset):
    scene,_,_ = dataset[0]
    return scene


def main():
    dataset = Exp1Dataset("/databases/exp1.hdf5")
    scene = get_scene(dataset)
    profile_scene(scene)

if __name__ == '__main__':
   main()
