export run_exp1_trial

"""
Simple example of inferring density and friction with no visual prior
"""

Gen.load_generated_functions()

function rejuv(trace)
    (new_trace, _) = Gen.mh(trace, gibbs_step, tuple())
    return new_trace
end

function run_inference(args, init_obs, init_args,
                       observations, iter::Int = 100)

    latents = Dict( :x => x -> :x )
    query = Gen_Compose.SequentialQuery(latents, #bogus for now
                                        markov_generative_model,
                                        init_args,
                                        init_obs,
                                        args,
                                        observations)


    # -----------------------------------------------------------
    # Define the inference procedure
    # In this case we will be using a particle filter
    #
    # Additionally, this will be under the Sequential Monte-Carlo
    # paradigm.
    n_particles = iter
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    @time sequential_monte_carlo(procedure, query)
    # @profview sequential_monte_carlo(procedure, query)
end

function run_exp1_trial(dpath::String, idx::Int, particles::Int,
                        out::String="")
    d = galileo_ramp.Exp1Dataset(dpath)
    (scene, state, _) = get(d, idx)
    n = size(state["pos"], 1)
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = scene["initial_pos"]["A"]
    cm[:initial_state => 2 => :init_pos] = scene["initial_pos"]["B"]
    obs = Vector{Gen.ChoiceMap}(undef, n)
    for t = 1:n
        tcm = choicemap()
        addr = :chain => t => :positions
        tcm[addr] = state["pos"][t, :, :]
        obs[t] = tcm
    end

    params = Params(2, [scene["objects"]["A"],
                        scene["objects"]["B"]])
    args = [(t, params) for t in 1:n]

    results = run_inference(args, cm, (0, params),
                            obs, particles)
    save_state(results, out)
    return results
end
