#!/usr/bin/env python-jl

import os
import sys
import dask
import numpy as np
from multiprocessing import set_executable
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


import distributed
from dask_jobqueue import SLURMCluster


# path to trial dataset
dataset = "/databases/exp1.hdf5"
# path to human responses
responses = "/databases/exp1_avg_human_responses.csv"

INIT_JULIA = True

def eval_trial(obs_noise, particles, trial):
    """ Runs inference for a given trial"""
    # load julia modules for worker
    import galileo_ramp.execute
    gr = galileo_ramp.execute.initialize(INIT_JULIA)
    return gr.evaluation(obs_noise, particles, dataset, trial)

def merge(results):
    """ Merges inference runs and returns RMSE """
    import galileo_ramp.execute
    gr = galileo_ramp.execute.initialize(INIT_JULIA)
    return gr.merge_evaluation(results, responses)


trials = list(range(210))
# trials = [0,1,120,121]

def f(obs_noise, particles, client):
    """ The black box function that returns RMSE """
    g = lambda t: eval_trial(obs_noise, particles, t)
    tasks = client.map(g, trials, pure = False)
    results = client.gather(tasks)
    deviance = client.submit(merge, results)
    return deviance.result() * -1.0

def initialize_dask(n):

    slurm = n > 1
   
    if not slurm:
        # must initialize julia globalyy
        import galileo_ramp.execute
        galileo_ramp.execute.initialize()
        global INIT_JULIA
        INIT_JULIA = False
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = 1,
                                           threads_per_worker = 1,
                                           processes = False,
        )
                                           # python = "python-jl")

    else:
        n = min(500, n)
        path = os.path.abspath(__file__)
        path_splits = path.split(os.path.sep)
        root = os.path.join(*path_splits[:-3])
        py = os.path.join('/'+root, 'run.sh python')
        params = {
            'death_timeout': 200,
            'nanny' : False,
            'python' : py,
            'cores' : 1,
            'memory' : '2GB',
            'walltime' : '0-6',
            'processes' : 1,
            'job_extra' : [
                '--partition short',
                # '--array 0-{0:d}'.format(n - 1),
                '--requeue',
                # '--output "/dev/null"'
                # ('--output ' + os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out')),
            ],
            #'env_extra' : [
            #    'JOB_ID=${SLURM_ARRAY_JOB_ID%;*}_${SLURM_ARRAY_TASK_ID%;*}',

                # 'source /etc/profile.d/modules.sh',
                # 'cd {0!s}'.format(CONFIG['PATHS', 'root']),
            # ]
        }
        cluster = SLURMCluster(**params)
        print(cluster.job_script())
        cluster.scale(n)

    print(cluster.dashboard_link)
    return distributed.Client(cluster)

def main():
    # print(sys.executable)
    set_executable('/project/.pyenv/bin/python-jl')
    client = initialize_dask(len(trials))

    # run function once for julia JIT
    f(0.1, 1, client)

    # partial application of fitness function
    def black_box(obs_noise = 0.1, particles = 1):
        return f(obs_noise, int(particles), client)

    # Bounded region of parameter space
    pbounds = {
        'obs_noise': (0.001, 0.1),
        'particles': (1, 100),
    }
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=pbounds,
        verbose=2,
        random_state=1)

    logger = JSONLogger(path="/traces/exp1_pf_bo_logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    client.close()

if __name__ == "__main__":
    main()


