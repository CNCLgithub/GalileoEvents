
using Random:shuffle
using Base.Iterators
using AsyncManager
using Distributed
using BayesianOptimization
using GaussianProcesses
using Distributions

# adding workers
# addprocs(2);
path = pwd()
# # TODO: Do I need `path`?
manager = AsyncSlurmManager(120, path;
                            partition = "short",
                            time = 120)

@sync addprocs(manager; exename = "./run.sh julia",
         dir = path)

const human_responses = "/databases/exp1_avg_human_responses.csv"

@everywhere begin
    using GalileoRamp, DataFrames

    const exp1_dataset = "/databases/exp1.hdf5"

    # evaluation on individual trial
    function run_trial(trial::Int; kwargs...)
        # random function for now
        df = GalileoRamp.evaluation(exp1_dataset, trial;
                                    kwargs...)
    end
end

"""
Performs k-fold cross-validation over partitions of
matched pairs. 
"""
function objective(x)

    args = (obs_noise = x[1],
            prior_width = x[2],
            particles = 100,
            chains = 5,
            bo_ret = true)

    n = 120
    k = 40
    test = shuffle(collect(1:n) .<= k
    trials = findall(.!(test)) .- 1 # shift to 0 base
    collected = @sync pmap(x->run_trial(x;args...), trials;
                           on_error=identity)

    merged = merge_evaluation(collected, human_responses)
    merged = @linq merged |>
       where(:scene .< n) |>
       transform(:test = repeat(test, inner = 4)          

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
                   logNoise = 0., # assume some noise in observations (due to approximation accuracy)
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.


# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
modeloptimizer = MAPGPOptimizer(every = 10, noisebounds = [-4, 3],       # bounds of the logNoise
                                kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 40)

opt = BOpt(objective,
           model,
           ExpectedImprovement(),                   # type of acquisition
           modeloptimizer,
           [0.01, 0.01], [0.2, 0.99],                     # lowerbounds, upperbounds
           repetitions = 2,                          # evaluate the function for each input n times
           maxiterations = 100,                      # evaluate at n input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

result = boptimize!(opt)
println(result)

#for i in workers()
#    rmprocs(i)
#end
