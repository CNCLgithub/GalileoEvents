
using Base.Iterators
using AsyncManager
using Distributed
using BayesianOptimization
using GaussianProcesses
using Distributions
using DataFrames

# adding workers
addprocs(1);
# path = pwd()
# project_path = dirname(dirname(path))
# # TODO: Do I need `path`?
# manager = AsyncSlurmManager(2, path;
#                            partition = "short",
#                            time = 120)
# addprocs_slurm(manager; exename = "./run.sh julia",
#                dir = project_path)

const human_responses = "/databases/exp1_avg_human_responses.csv"

@everywhere begin
    using GalileoRamp

    const exp1_dataset = "/databases/exp1.hdf5"

    # evaluation on individual trial
    function run_trial(channel, trial; kwargs...)
        # random function for now
        df = GalileoRamp.evaluation(exp1_dataset, trial;
                                    kwargs...)
        println(df)
        put!(channel, df)
    end
end

function objective(x)

    args = (obs_noise = x[1],
            prior_width = x[2],
            particles = 10,
            chains = 1,
            bo_ret = true)

    display(args)
    n = 10
    # Cribbed from https://white.ucc.asn.au/2018/07/14/Asynchronous-and-Distributed-File-Loading.html
    results = Channel(ctype=DataFrame, csize=n) do results

        remote_ch = RemoteChannel(()->results)

        @sync @distributed for idx = 1:n
            trial = idx - 1
            run_trial(remote_ch, trial; args...)
        end

        collect(take(results, n))
        # c_pool = CachingPool(workers())
        # println(c_pool)
        # futures = map(1:n) do idx
        #     trial = idx - 1
        #     remotecall(run_trial, c_pool, remote_ch, trial; args...)
        # end
        # for f in futures
        #     wait(f)
        # end
        # clear!(c_pool)
    end

    println(results)

    merged = merge_evaluation(results, human_responses)
    rmse = GalileoRamp.fit_pf(merged)

    println("input: $x")
    println("objective: $rmse")

    return rmse
end


# Choose as a model an elastic GP with input dimensions 2.
# The GP is called elastic, because data can be appended efficiently.
model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(0.),
                   kernel = SEArd([0., 0.], 5.),
                   logNoise = 0.,
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])


# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       # bounds of the logNoise
                                kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 40)

opt = BOpt(objective,
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,
           [0.01, 0.2], [0.01, 0.9],                     # lowerbounds, upperbounds
           repetitions = 5,                          # evaluate the function for each input 5 times
           maxiterations = 10,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

result = boptimize!(opt)
display(result)

clear!(workers())
for i in workers()
    rmprocs(i)
end
