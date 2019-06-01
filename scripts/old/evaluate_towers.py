#!/bin/python3
""" Evaluates a batch of blocks for stimuli generation.

Performs a series of analysis.

1. Physics differentials. (Optional)

Given an exhaustive dataframe (all configurations for a given set of towers),
determine how each configuration changes physical stability and direction of
falling.

2. Computes a histogram over the 2-D differential space.

Each configuration across towers is plotted where each dimension
is normalized by the average and variance of the entire pool.
"""

import os
import copy
import json
import argparse

import numpy as np
import pandas as pd

from config import Config
from utils import plot

CONFIG = Config()


def idxquantile(s, q=0.5, *args, **kwargs):
    """ Returns the specified quantile.
    """
    qv = s.quantile(q, *args, **kwargs)
    return (s.sort_values()[::-1] <= qv).idxmax()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_2vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
     12318300_0       3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_2vec(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')
    parser.add_argument('--quantiles', type = float, nargs = '+',
                        default = [0.25, 0.5, 0.75],
                        help = 'Quantiles to extract.')

    args = parser.parse_args()

    src = os.path.join(CONFIG['data'], args.src + '_stability.json')

    with open(src, 'r') as f:
        stats = json.load(f)

    df = []
    for tower in stats:
        # print(tower)
        results = stats[tower]
        original = results['template']
        for struct in results:
            dis = np.linalg.norm(np.array(results[struct]['positions']) -\
                                 np.array(results['template']['positions']))
            m = {k : v for k,v in results[struct].items() if k != 'positions'}
            df.append({'tower' : tower, 'id' : struct, 'l2' : dis, **m})

    df = pd.DataFrame(df)
    for tower, g in df.groupby('tower'):
        orig = g[g['id'] == 'template']
        df.loc[g.index, 'instability_diff'] = g['instability'] - \
                                              np.array(orig['instability'])
        df.loc[g.index, 'mag_diff'] = g['mag'] - \
                                              np.array(orig['mag'])
        df.loc[g.index, 'angle_diff'] = g['angle'] - \
                                              np.array(orig['angle'])
        # df.loc[g.index, 'l2_dis'] = np.linalg.norm(g['positions'] - \
        #                                            np.array(orig['positions']))
        if len(g) != 2:
            print(g)

    print(df.sort_values(['tower', 'instability_diff']))
    out = os.path.join(CONFIG['data'], args.src + '_differentials.csv')
    df.sort_values(['tower', 'instability_diff']).to_csv(out)
    df_filtered = df[df.id != 'template']


    quantiles = []
    for dim in ['instability_diff', 'l2', 'angle_diff']:
        data = []
        radius = np.max(df.mag)
        print(df_filtered[dim].abs().describe())
        for quant in args.quantiles:
            row = idxquantile(df_filtered[dim].abs(), quant)
            dat = df_filtered.loc[row]
            orig = df[(df.id == 'template') & (df.tower == dat.tower)]
            data.append((dat, orig))
            path = os.path.join(CONFIG['data'], args.src,
                                dat.tower +'_orig.json')
            quantiles.append(
                {
                    'dim': dim,
                    'quant' : '{0:d}'.format(int(quant*100)),
                    'path' : path,
                    'id' : dat.id,
                    'angle' : float(dat.angle),
                    'o_angle': float(orig.angle),
                })

        if dim == 'l2':
            out = os.path.join(CONFIG['data'], args.src + '_l2_angle_diff.png')
            plot.plot_direction_diff(data, args.quantiles, radius, out)
        if dim == 'angle_diff':
            out = os.path.join(CONFIG['data'], args.src + '_angle_diff.png')
            plot.plot_direction_diff(data, args.quantiles, radius, out)

        out = os.path.join(CONFIG['data'], args.src + '_' + dim + '_hist.png')
        plot.plot_tower_diff(df_filtered, out, dim)

    out = os.path.join(CONFIG['data'], args.src + '_quantiles.json')
    with open(out, 'w') as f:
        json.dump(quantiles, f, indent = 4)

    heavy = df[df['id'].apply(lambda x: 'H' in x)].tower
    light = df[df['id'].apply(lambda x: 'L' in x)].tower
    # out = os.path.join(CONFIG['data'], args.src + '_trials.json')
    # d = {
    #     'heavy' : list(np.random.choice(heavy, size = 30)),
    #     'light' : list(np.random.choice(light, size = 30))

    # }
    # with open(out, 'w') as f:
    
    print(len(heavy))
    print(len(light))


if __name__ == '__main__':
   main()
