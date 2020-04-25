export static_inference, seq_inference

using Base.Filesystem

"""
"""


######################################################################
# Helpers
######################################################################

function load_trial(dpath::String, idx::Int, obs_noise::Float64)
    d = galileo_ramp.Exp1Dataset(dpath)
    (scene, state, _) = get(d, idx)

    n = size(state["pos"], 1)
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = scene["initial_pos"]["A"]
    cm[:initial_state => 2 => :init_pos] = scene["initial_pos"]["B"]
    objects = scene["objects"]
    cm[:object_physics => 2 => :material] = 2
    cm[:object_physics => 2 => :congruent] = true
    cm[:object_physics => 2 => :density] = objects["B"]["physics"]["density"]
    cm[:object_physics => 2 => :friction] = objects["B"]["physics"]["lateralFriction"]
    cm[:object_physics => 2 => :restitution] = objects["B"]["physics"]["restitution"]
    cm[:object_physics => 1 => :material] = findfirst(k -> k == objects["A"]["appearance"],
                                                      mat_keys)
    cm[:object_physics => 1 => :friction] = objects["A"]["physics"]["lateralFriction"]
    cm[:object_physics => 1 => :restitution] = objects["A"]["physics"]["restitution"]
    # cm[:object_physics => 1 => :congruent] = true

    obs = Vector{Gen.ChoiceMap}(undef, n)
    for t = 1:n
        tcm = choicemap()
        addr = :chain => t => :positions
        tcm[addr] = state["pos"][t, :, :]
        # keep table at gt
        # tcm[:chain => t => :physics => 2 => :persistence => :congruent] = true
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

function resume_pf(chain_path)
    chain = extract_chain(chain_path)
    addr = :object_physics => 1 => :density
    temp = chain["weighted"][:ramp_density]
    t = size(temp, 1)
    estimates = Dict(addr => temp)
    return (t, estimates)
end

######################################################################
# LatentMaps
######################################################################

function extract_pos(t)
    state,graph,belief = last(get_retval(t))
    all_pos = reshape(state[1, :, :], (1,2,3))
    reshape(all_pos, (1,1,2,3))
end

function extract_collision(t)
    i,params = get_args(t)
    addr = :chain => i => :graph => :collision
    d = Vector{Float64}(undef, 1)
    d[1] = Gen.get_choices(t)[addr]
    reshape(d, (1,1,1))
end

function extract_sliding(t, obj::Int)
    i,params = get_args(t)
    addr = :chain => i => :graph => :self_edges => obj => :sliding
    d = Vector{Float64}(undef, 1)
    d[1] = Gen.get_choices(t)[addr]
    reshape(d, (1,1,1))
end

function extract_phys(t, feat)
    state,graph,belief = last(get_retval(t))
    d = Vector{Float64}(undef, 1)
    # println(belief[1])
    d[1] = belief[1][feat]
    reshape(d, (1,1,1))
end

const seq_latent_map = LatentMap(Dict(
    :position => extract_pos,
    # :collision => extract_collision,
    :ramp_sliding => t -> extract_sliding(t, 1),
    :table_sliding => t -> extract_sliding(t, 2),
    :ramp_density => t -> extract_phys(t, "density"),
    :ramp_congruent => t -> extract_phys(t, "congruent"),
    # :ramp_friction => t -> extract_phys(t, :friction),
))
const light_seq_map = LatentMap(Dict(
    :ramp_density => t -> extract_phys(t, "density"),
))

######################################################################
# Inference Calls
######################################################################

"""
Sequentail estimation of markovian posterior
"""
function seq_inference(dpath::String, idx::Int, particles::Int,
                       obs_noise::Float64;
                       resume::Bool = false,
                       out::Union{String, Nothing} = nothing,
                       bo::Bool = false)
    params, constraints, obs = load_trial(dpath, idx, obs_noise)
    nt = length(obs)
    args = [(t, params) for t in 1:nt]

    lm = bo ? light_seq_map : seq_latent_map
    query = Gen_Compose.SequentialQuery(lm,
                                        cp_generative_model,
                                        (0, params),
                                        constraints,
                                        args,
                                        obs)

    ess = particles * 0.5
    proc= PopParticleFilter(particles,
                            ess,
                            nothing,
                            tuple(),
                            cp_rejuv,
                            # nothing,
                            false)

    buffer_size = bo ? 120 : 40
    out = bo ? nothing : out

    if ((isnothing(out) || isfile(out)) && resume)
        leftoff, choices  = resume_pf(out)
        println("Resuming trace $out at $(leftoff+1)")
        results = sequential_monte_carlo(proc, query, leftoff + 1, choices,
                                         path = out,
                                         buffer_size = buffer_size)
    else
        println("New trace at $out")
        results = sequential_monte_carlo(proc, query,
                                         path = out,
                                         buffer_size = buffer_size)

    end

    physics.physics.clear_trace(params.client)
    return results
end
