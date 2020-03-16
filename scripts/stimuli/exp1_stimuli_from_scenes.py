#!/usr/bin/env python3

""" Creates stimuli movies from image renderings.

Uses a timing file containing time points of interest
for each scene to be generated.

"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from galileo_ramp.exp1_dataset import Exp1Dataset

data_to_copy = ['appearance', 'shape', 'volume']
def extract_scene_data(scene):
    ramp_obj = scene['objects']['A']
    data = {k:ramp_obj[k] for k in data_to_copy}
    data.update(ramp_obj['physics'])
    data['pos_x_ramp'] = ramp_obj['position'][0]
    data['pos_x_table'] = scene['objects']['B']['position'][0]
    data['init_pos_ramp'] = scene['initial_pos']['A']
    data['init_pos_table'] = scene['initial_pos']['B']
    return data

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type = str,
                        default = '/databases/exp1.hdf5',
                        help = 'Path to dataset')
    parser.add_argument('--n_cond', type = int,
                        default = 8)
    args = parser.parse_args()

    dataset = Exp1Dataset(args.dataset)

    out = '/movies/trials'
    if not os.path.isdir(out):
        os.mkdir(out)

    trial_data = []
    idx = 0

    for t in range(len(dataset)):
        (scene, _, time_points) = dataset[t]
        out_path = os.path.join(out, str(t))
        # stims_from_scene(i, out_path, *cond, args.pad)

        if t < 120:
            tidx = int(np.floor(t / 2))
            con = (t % 2) == 0
            tpe = "matched"
        else:
            tidx = t - 60
            con = True
            tpe = "control"
           

        scene_data = extract_scene_data(scene)
        scene_data.update({
            'scene' : tidx,
            'congruent' : con,
            'type' : tpe
            })

        for c,time in enumerate(time_points):
            path = '{0:d}_t-{1:d}.mp4'.format(t, c)
            trial_datum = {
                'idx' : idx,
                'path' : path,
                'cond' : c,
                'time' : time
            }
            trial_datum.update(scene_data)
            trial_data.append(trial_datum)
            idx += 1


    df = pd.DataFrame.from_records(trial_data)
    df.to_csv(out + '/trial_data.csv', index = False)

    condition_list = []
    i = 0
    for cond in range(args.n_cond):
        clist = []
        for t in range(150):
            idx = int((i + cond) % args.n_cond)
            if t < 60:
                if idx < 4:
                    path = '{0:d}_t-{1:d}.mp4'.format(t*2, idx % 4)
                else:
                    path = '{0:d}_t-{1:d}.mp4'.format(t*2 + 1, idx % 4)

            else:
                path = '{0:d}_t-{1:d}.mp4'.format(60 + t, idx % 4)
            clist.append(path)
            i += 1
        condition_list.append(clist)

    out_path = os.path.join(out, 'condlist.json')
    with open(out_path, 'w') as f:
        json.dump(condition_list, f, indent = 2)



if __name__ == '__main__':
    main()
