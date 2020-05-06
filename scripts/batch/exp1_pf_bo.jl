
using AsyncManager
using Distributed

# adding workers
addprocs(2);
# path = pwd()
# project_path = dirname(dirname(path))
# # TODO: Do I need `path`?
# manager = AsyncSlurmManager(2, path;
#                            partition = "short",
#                            time = 120)
# addprocs_slurm(manager; exename = "./run.sh julia",
#                dir = project_path)


@everywhere begin
    using BayesianOptimization
    using GalileoRamp

    const exp1_dataset = "/databases/exp1.hdf5"

    # evaluation on individual trial
    function evaluation(trial; kwargs...)
        # random function for now
        GalileoRamp.evaluation(exp1_dataset, trial;
                               kwargs...)
    end
end

function objective(x)

    args = (obs_noise = x[1],
            prior_width = x[2],
            particles = 10,
            chains = 10,
            bo_ret = true)

    results = SharedArray{DataFrame}(120)
    @distributed for trial=0:119
        results[i+1] = evaluation(trial; args...)
    end

    merged = merge_evaluation(results, human_responses)
    rmse = fit_pf(merged)

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
           [0., 0.1], [0., 1.],                     # lowerbounds, upperbounds
           repetitions = 5,                          # evaluate the function for each input 5 times
           maxiterations = 100,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

result = boptimize!(opt)
display(result)

for i in workers()
    rmprocs(i)
end
