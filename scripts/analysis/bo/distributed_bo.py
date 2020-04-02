#!/usr/bin/env python
from sklearn.metrics import mean_squared_error
from math import sqrt

def eval_trial(obs_noise, particles, trial):
    return gr.evaluation(obs_noise, particles, trial)

def f(obs_noise, particles):
    g = lambda t: eval_trial(obs_noise, particles, t)
    tasks = client.map(g, np.arange(210), pure = False)
    results = client.gather(tasks)
    slope, intercept, _, _, _ = stats.linregress(results[60:], responses[60:])
    preds = results[:60] * slope + intercept
    return sqrt(mean_squared_error(responses[:60], preds))


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
