using ClusterManagers, Distributed

# initialize master
#master_server = start_master(n_workers)
# start client
#client = start(master_server)

# adding workers
addprocs_slurm(20, p="short", t="00:20:00", exename="julia.sh", dir="/gpfs/milgram/project/yildirim/eivinas/dev/galileo-ramp")


@everywhere begin
using BayesianOptimization, GaussianProcesses, Distributions
using SharedArrays

# evaluation on individual trial
function evaluation(measurement_noise, num_particles, trial)
    # random function for now
    return x[1]^2 + x[2]^3 + rand(Normal(0,1))
end
    
end


# full model evaluation
# x[1] - measurement_noise
# x[2] - num_particles
function full_evaluation(x)

    measurement_noise = x[1]
    num_particles = x[2]

    #tasks = map(t -> inference(t, x), trials)
    
    results = SharedArray{Float64}(100)
    @distributed for trial=1:100
        #results[i] = remotecall_fetch(evaluation, workers()[i], measurement_noise, num_particles, i)
        results[i] = evaluation(measurement_noise, num_particles, trial)
    end

    #results = client.collect(tasks)
    
    # mean value for now
    # should be correlation(results, behavioural_data) 
    result = mean(results)

    println("input: $x")
    println("result: $result")

    return result
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

opt = BOpt(full_evaluation,
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

for i in workers()
    rmprocs(i)
end
