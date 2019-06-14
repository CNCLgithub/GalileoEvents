"""
Defines tools for manipulating trace data
"""
import h5py
import argparse
import numpy as np

from ..utils import hdf5

def extract_map(chain):
    idxs = np.argmax(chain['scores'], axis = 1)
    es = chain['estimates']
    maps = es[np.arange(es.shape[0]), idxs]
    scores = chain['scores']
    ss = scores[np.arange(scores.shape[0]), idxs]
    return maps, ss

def extract_chains(trace, maps = True):
    with h5py.File(trace) as f:
        result = hdf5.to_dic(f)
    n_chains = len(result['chains'])
    xs = result['xs']
    n_latents = result['chains']['0']['estimates'].shape[2]
    if maps:
        estimates = np.zeros((n_latents, len(xs), n_chains))
        scores = np.zeros((len(xs), n_chains))
        for i,c in enumerate(result['chains']):
            es, ss = extract_map(result['chains'][c])
            estimates[:, :, i] = es.T
            scores[:, i] = ss
    else:
        n_particles = result['chains']['0']['estimates'].shape[1]
        estimates = np.zeros((n_latents, len(xs), n_chains * n_particles))
        scores = np.zeros((len(xs), n_chains * n_particles))
        for i,c in enumerate(result['chains']):
            es = result['chains'][c]['estimates']
            es = np.transpose(es, (2, 0, 1))
            ss = result['chains'][c]['scores']
            estimates[:, :, i*n_particles:(i+1)*n_particles] = es
            scores[:, i*n_particles:(i+1)*n_particles] = ss

    return estimates, scores, result
