export static_inference, seq_inference

"""
"""

function rejuv(trace)
    # return trace
    (new_trace, accepted) = Gen.mh(trace, exp1_gibbs, tuple())
    return new_trace
end

function run_inference(query::StaticQuery, proc, out::Union{String, Nothing})
    static_monte_carlo(proc, query, path = out)
end
function run_inference(query::SequentialQuery, proc, out::Union{String, Nothing})
    sequential_monte_carlo(proc, query, path = out)
end

function load_trial(dpath::String, idx::Int, obs_noise::Float64)
    d = galileo_ramp.Exp1Dataset(dpath)
    (scene, state, _) = get(d, idx)

    n = size(state["pos"], 1)
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = scene["initial_pos"]["A"]
    cm[:initial_state => 2 => :init_pos] = scene["initial_pos"]["B"]
    objects = scene["objects"]
    cm[:object_physics => 2 => :density] = objects["B"]["physics"]["density"]
    cm[:object_physics => 2 => :friction] = objects["B"]["physics"]["lateralFriction"]
    cm[:object_physics => 2 => :restitution] = objects["B"]["physics"]["restitution"]
    cm[:object_physics => 1 => :friction] = objects["A"]["physics"]["lateralFriction"]
    cm[:object_physics => 1 => :restitution] = objects["A"]["physics"]["restitution"]
    # FOR DEBUGGING
    # cm[:object_physics => 1 => :density] = objects["A"]["physics"]["density"]
    # cm[:object_physics => 1 => :density] = 2.0

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
    params = Params(obj_prior, init_pos, scene, obs_noise, cid)
    return (params, cm, obs)
end

function extract_pos(t)
    ret = Gen.get_retval(t)
    all_pos = [reshape(state[1, :, :], (1,2,3)) for state in ret]
    all_pos = vcat(all_pos...)
    reshape(all_pos, (1, size(all_pos)...))
end

function extract_phys(t, feat)
    d = Vector{Float64}(undef, 1)
    d[1] = Gen.get_choices(t)[:object_physics => 1 => feat]
    reshape(d, (1,1,1))
end

const static_latent_map = LatentMap(Dict(
    :ramp_pos => extract_pos,
    :ramp_density => t -> extract_phys(t, :density),
))
const seq_latent_map = LatentMap(Dict(
    :ramp_pos => t -> reshape(extract_pos(t)[:, end, :, :], (1,1,2,3)),
    :ramp_density => t -> extract_phys(t, :density),
))

"""
Static estimation of full posterior
P(ramp_density | O1,..,ON, State_0)
"""
function static_inference(dpath::String, idx::Int, samples::Int,
                          obs_noise::Float64;
                          out::Union{String, Nothing} = nothing)
    params, constraints, obs = load_trial(dpath, idx, obs_noise)
    for o in obs
         for (addr, submap) in get_submaps_shallow(o)
             set_submap!(constraints, addr, submap)
         end
    end
    args = (length(obs), params)
    query = Gen_Compose.StaticQuery(static_latent_map,
                                    markov_generative_model,
                                    args,
                                    constraints)
    proc = MetropolisHastings(samples, rejuv)
    run_inference(query, proc, out)
    physics.physics.clear_trace(params.client)
end

"""
Sequentail estimation of markovian posterior
"""
function seq_inference(dpath::String, idx::Int, particles::Int,
                       obs_noise::Float64;
                       out::Union{String, Nothing} = nothing)
    params, constraints, obs = load_trial(dpath, idx, obs_noise)
    args = [(t, params) for t in 1:length(obs)]
    query = Gen_Compose.SequentialQuery(seq_latent_map,
                                        markov_generative_model,
                                        (0, params),
                                        constraints,
                                        args,
                                        obs)
    ess = particles * 0.5
    proc= ParticleFilter(particles,
                         ess,
                         rejuv)
    run_inference(query, proc, out)
    physics.physics.clear_trace(params.client)
end
