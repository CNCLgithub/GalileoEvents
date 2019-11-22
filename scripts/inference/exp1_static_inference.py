#!/usr/bin/env python3
"""
Computes MH-MCMC over Exp1 scenes
"""
import os
import glob
import json
import argparse
import datetime
import numpy as np

from galileo_ramp.utils import config
from galileo_ramp.inference.execute import initialize

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']
module_path = os.path.join(root, 'inference',
                           'queries', 'exp1_static_inference.jl')
inference = initialize(module_path)

def run_search(scene_json, scene_pos, out):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    positions = np.load(scene_pos)
    distances = np.linalg.norm(positions[1] - positions[0], axis = -1)
    positions = np.transpose(positions, (1, 0, 2))
    contacted = (distances[1:] - distances[:-1]) > 0
    collision_frame = np.where(contacted)[0][0]
    before = collision_frame - 1
    just_after = collision_frame  + 12
    half_way = int((len(positions) + collision_frame) / 2)
    full = len(positions)
    time_points = [before, just_after, half_way, full]
    print(time_points)
    with open(scene_json, 'r') as f:
        scene_data = json.load(f)['scene']

    for i,t in enumerate(time_points):
        out_path = "{0!s}_{1:d}".format(out, i)
        inference(scene_data, positions[:t, :, :], out_path)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trial', type = str, help = 'path to scene file')
    # misc
    parser.add_argument('--out', type = str, help = 'directory to save traces',
                        default =  'exp1_static_inference')

    args = parser.parse_args()

    print('Initializing inference run')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = os.path.basename(os.path.splitext(args.trial)[0])
    if not os.path.isdir(out):
        try:
            os.mkdir(out)
        except:
            print('{0!s} already exists'.format(out))

    out_path = os.path.join(out, trial_name)
    print('Saving results in {0!s}'.format(out))
    position_file = args.trial.replace('.json', '_pos.npy')
    if os.path.isfile(out_path + '_trace.csv'):
        print('Inference already complete')
    else:
        run_search(args.trial, position_file, out_path)


if __name__ == '__main__':
    main()
