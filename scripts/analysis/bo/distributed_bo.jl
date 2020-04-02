using ClusterManagers
using Distributed
using Base.Filesystem

# initialize master
#master_server = start_master(n_workers)
# start client
#client = start(master_server)

# adding workers
addprocs_slurm(20, p="short", t="00:20:00", exename="julia.sh", dir=cwd())


@everywhere begin
using BayesianOptimization, GaussianProcesses, Distributions
using SharedArrays

# evaluation on individual trial
function evaluation(obs_noise, num_particles, trial)
    # out = "/traces/exp1_pf_bo_p_$(particles)_n_$(obs_noise)"
    # clean up previous run
    # isfile(out) || rm(out)
    chain = seq_inference(dataset, trial, particles, obs_noise;
                          bo = true)
    # returns tibble of: | :t | :ramp_density_mean | :log_score_mean
    digest_pf_trial(chain)
end


# full model evaluation
# x[1] - measurement_noise
# x[2] - num_particles
function full_evaluation(x)

    measurement_noise = x[1]
    num_particles = x[2]

    #tasks = map(t -> inference(t, x), trials)

    # | :scene | :t | :ramp_density_mean | :log_score_mean
    results = DataFrame()
    @distributed for trial=1:210
        df = evaluation(measurement_noise, num_particles, trial)
        if trial < 120
            df.scene = floor(trial/2)
            df.congruent = (trial % 2) == 0
        else
            df.scene = trial - 60
            df.congruent = true
        end
        results = vcat(results, df)
    end

    results = join(results, human_respones, on = [:scene, :congruent])
    rmse = digest_pf_trial(results)
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
