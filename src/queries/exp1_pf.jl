export run_exp1_trial

"""
Runs inference on Exp1 trials
"""

function rejuv(trace)
    (new_trace, _) = Gen.mh(trace, gibbs_step, tuple())
    return new_trace
end

function run_inference(args, init_obs, init_args,
                       observations, n_particles::Int = 1,
                       out::Union{String,Nothing} = nothing)

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
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    sequential_monte_carlo(procedure, query, path = out)
end

function run_exp1_trial(dpath::String, idx::Int, particles::Int,
                        obs_noise::Float64, out::Union{String, Nothing})
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

    obj_prior = [scene["objects"]["A"],
                 scene["objects"]["B"]]
    init_pos = [scene["initial_pos"]["A"],
                scene["initial_pos"]["B"]]
    cid = physics.physics.init_client(direct = true)
    params = Params(obj_prior, init_pos, obs_noise, cid)
    args = [(t, params) for t in 1:n]

    results = run_inference(args, cm, (0, params),
                            obs, particles, out)
    physics.physics.clear_trace(cid)
    return results
end
