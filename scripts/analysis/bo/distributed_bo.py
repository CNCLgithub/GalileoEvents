#!/usr/bin/env python-jl

import numpy as np
from math import sqrt
from galileo_ramp import execute
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error


# path to human responses
responses = "/databases/exp1_avg_human_responses.csv"
# load julia modules for master
gr = execute.initialize()

# Bounded region of parameter space
pbounds = {'obs_noise': (0.001, 0.8),
           'particles': (1, 300)}

def eval_trial(obs_noise, particles, trial):
    # load julia modules for worker
    gr = execute.initialize()
    return gr.evaluation(obs_noise, particles, trial)

def f(obs_noise, particles):
    g = lambda t: eval_trial(obs_noise, particles, t)
    tasks = client.map(g, np.arange(210), pure = False)
    results = client.gather(tasks)
    rmse = gr.merge_evaluation(results, responses)
    return rmse

def main():

    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        verbose=2,
        random_state=1)

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

if __name__ == "__main__":
    main()

def initialize_dask(n, slurm = False):

    if not slurm:
        cores =  len(os.sched_getaffinity(0))
        cluster = distributed.LocalCluster(n_workers = cores,
                                           threads_per_worker = 1)

    else:
        n = min(500, n)
        py = os.path.join(root, '/run.sh python-jl')
        params = {
            'python' : py,
            'cores' : 1,
            'memory' : '2GB',
            'walltime' : '1-0',
            'processes' : 1,
            'job_extra' : [
                '--partition scavenge',
                '--array 0-{0:d}'.format(n - 1),
                '--requeue',
                # '--output "/dev/null"'
                # ('--output ' + os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out')),
            ],
            'env_extra' : [
                'JOB_ID=${SLURM_ARRAY_JOB_ID%;*}_${SLURM_ARRAY_TASK_ID%;*}',
                'source /etc/profile.d/modules.sh',
                'cd {0!s}'.format(CONFIG['PATHS', 'root']),
            ]
        }
        cluster = SLURMCluster(**params)
        print(cluster.job_script())
        cluster.scale(1)

    print(cluster.dashboard_link)
    return distributed.Client(cluster)
