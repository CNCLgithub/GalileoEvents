using ClusterManagers, Distributed

# initialize master
#master_server = start_master(n_workers)
# start client
#client = start(master_server)

# adding workers
addprocs_slurm(2, p="short", t="00:5:00", exename="julia.sh", dir="/gpfs/milgram/project/yildirim/eivinas/dev/galileo-ramp")

"""
for i in workers()
    host, pid = fetch(@spawnat i (gethostname(), getpid()))
end
"""

@everywhere begin
using BayesianOptimization, GaussianProcesses, Distributions

# evaluation on individual trial
function evaluation(trial)
    # random number for now
    return rand(Normal(0,1))
end
    

end


function g(x)
    #tasks = map(t -> inference(t, x), trials)
    results = Array{Float64}(undef, 2)
    for (i, worker) in enumerate(workers())
        println(workers())
        results[i] = fetch(remotecall(evaluation, worker, (i,)))
    end
    #results = client.collect(tasks)
    return mean(results)
    # return correlation(results, behavioural_data) 
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

opt = BOpt(g,
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,                        
           [-5., -5.], [5., 5.],                     # lowerbounds, upperbounds         
           repetitions = 5,                          # evaluate the function for each input 5 times
           maxiterations = 100,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

result = boptimize!(opt)


