#!/usr/bin/env python

""" Loads a scene json directly into pybullet
"""

import os
import json
import argparse
import numpy as np

from galileo_ramp.utils import config
from galileo_ramp.world.scene.ball import Ball
from galileo_ramp.world.scene.ramp import RampScene
from galileo_ramp.world.simulation import forward_model

CONFIG = config.Config()

def simulate_scene(src):
    """ Encapsulation for searching for interesting towers.

    Generates a random tower from the provided base and
    determine if it is an interesting tower.

    Returns:
    A tuple (passed, mut) where `passed` is `True` if the tower is interesting and
    `mut` is the mutation type (None if `passed == False`)
    """
    with open(src, 'r') as f:
        scene_data = json.load(f)['scene']

    s = forward_model.simulate(scene_data, 900, debug =True)
    print(np.sum(s[-1], axis = 0))

def main():

    parser = argparse.ArgumentParser(
        description = 'Loads a scene json directly into pybullet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str,
                        help = 'Path to json')
    args = parser.parse_args()

    src = os.path.join(CONFIG['PATHS', 'scenes'], args.src)
    simulate_scene(src)


if __name__ == '__main__':
   main()
