
export cp_generative_model

using LinearAlgebra:norm

## Generative Model + components

const material_ps = ones(3) ./ 3
const incongruent_mat = Dict( "density" => (0.01, 150.),
                              "lateralFriction" => (0.01, 0.99))

struct ObjectBelief
    congruent::Bool
    physical_latents::RigidBodyLatents
end

function cp_material_params(mat::String, w::Float64)
    dens_mu = density_map[mat]
    density_prior = (dens_mu * (1-w), dens_mu * (1+w))
    fric_mu = friction_map[mat]
    friction_prior = (fric_mu * (1-w), fric_mu * (1+w))
    # println("using $density_prior")
    return Dict("density" => density_prior,
                "lateralFriction" => friction_prior)
end

cp_material_params(i::Int, w::Float64) = cp_material_params(mat_keys[i], w)

@gen (static) function physics_prior(width::Float64)
    # Object appearance
    material = @trace(categorical(material_ps), :material)
    from_mat = cp_material_params(material, width)

    # Object physics -> appearance congruency
    congruent = @trace(bernoulli(0.9), :congruent)
    prior = congruent ? from_mat : incongruent_mat
    density = prior["density"]
    friction = prior["lateralFriction"]

    # Object physical properties
    dens = @trace(log_uniform(density[1], density[2]), :density)
    fric = @trace(log_uniform(friction[1], friction[2]), :friction)
    restitution = @trace(uniform(0.8, 1.0), :restitution)

    # Objects are not persistent untill the first collision
    physical_props = Dict("density" => dens,
                          "lateralFriction" => fric,
                          "restitution" => restitution,
                          "persistent" => false,
                          "congruent" => congruent,
                          "material" => material)
    return physical_props
end

map_object_prior = Gen.Map(physics_prior)

@gen (static) function state_prior()
    init_pos = @trace(uniform(0, 2), :init_pos)
    return init_pos
end

map_init_state = Gen.Map(state_prior)

function collision_probability(positions::T, dims::T) where T<:Matrix{Float64}
    dx = dims[:, 1] ./ 2.0
    l2 = norm(positions[1, 1] + dx[1] - positions[2,1] - dx[2])
    p = l2 < 0.35 ? 0.99 : 0.00
    # println("a_x $(positions[1,1]) l2 $(l2), p $(p)")
    return p
end
function sliding_probability(lin_vels::Vector{Float64})
    (abs(lin_vels[1]) > 1E-3) ? 0.95 : 0.01
end

@gen (static) function sliding(vels::Vector{Float64})
    sliding_p = sliding_probability(vels)
    slid = @trace(bernoulli(sliding_p), :sliding)
    return slid
end

map_sliding = Gen.Map(sliding)

@gen (static) function graph_kernel(prev_state::Array{Float64, 3},
                                    prev_cp::Bool,
                                    dims::Matrix{Float64})
    col_p = collision_probability(prev_state[1, :, :],
                                  dims)
    cp_p = prev_cp ? 0.0 : col_p
    cp = @trace(bernoulli(cp_p), :changepoint)
    args = [prev_state[4,1,:], prev_state[4,2,:]]
    slid = @trace(map_sliding(args), :self_edges)
    active_cp_edge = prev_cp | cp
    ret = (active_cp_edge, slid)
    return ret
end

parse_graph(t::Tuple) = first(t)

@gen function obj_persistence(prev_con::Bool, prev_dens::Float64,
                              material::Int, width::Float64)
    p = prev_con ? 0.9 : 0.1
    new_con = @trace(bernoulli(p), :congruent)
    if new_con == prev_con
        dens = prev_dens
    elseif prev_con
        # Con -> Incon
        dens = @trace(log_uniform(0.01, 150.0),  :density)
    else
        # Incon -> Con
        prior = cp_material_params(material, width)
        density = prior["density"]
        dens = @trace(log_uniform(density...), :density)
    end
    return (new_con, dens)
end

@gen function object_kernel(prev_phys::Dict{String, Float64},
                            col_edge::Bool,
                            width::Float64)

    congruent = Bool(prev_phys["congruent"])
    persistent = Bool(prev_phys["persistent"])
    material = Int(prev_phys["material"])
    prev_dens = prev_phys["density"]

    # if collision change => "persist"
    if col_edge & !persistent
        congruent, dens = @trace(obj_persistence(congruent, prev_dens,
                                                 material, width),
                                 :persistence)
        persistent = true
    else
        dens = prev_dens
    end
    prev_fric = prev_phys["lateralFriction"]
    physical_props = Dict{String, Float64}(
        "persistent" => persistent,
        "congruent" => congruent,
        "material" => material,
        "density" => dens,
        "lateralFriction" => prev_fric,
        "restitution" => prev_phys["restitution"])
    return physical_props
end

map_obj_kernel = Gen.Map(object_kernel)

function extract_dims(params::Params)
    dims = Matrix{Float64}(undef, 2, 3)
    for i = 1:length(params.object_prior)
        dims[i, :] = params.object_prior[i]["dims"]
    end
    return dims
end

@gen (static) function kernel(t::Int, prev::Tuple, params::Params)
    # prev_state, prev_graph, prev_phys = prev
    prev_cp = parse_graph(prev[2])
    dims = extract_dims(params)
    graph = @trace(graph_kernel(prev[1], prev_cp, dims), :graph)
    active_cp_edge = parse_graph(graph)
    cp_edge_change = !prev_cp & active_cp_edge
    arg1 = fill(cp_edge_change, 2)
    arg2 = fill(params.prior_width, 2)
    belief = @trace(map_obj_kernel(prev[3], arg1, arg2),
                    :physics)
    next_state = forward_step(prev[1], params, belief)
    pos = next_state[1, :, :]
    next_pos = @trace(Gen.broadcasted_normal(pos, params.obs_noise),
                      :positions)
    nxt = (next_state, graph, belief)
    return nxt
end

chain = Gen.Unfold(kernel)

@gen (static) function cp_generative_model(t::Int, params::Params)

    args = fill(params.prior_width, params.n_objects)
    objects = @trace(map_object_prior(args), :object_physics)
    initial_pos = @trace(map_init_state(args), :initial_state)
    i_state = initialize_state(params, objects, initial_pos)
    states = @trace(Gen.Unfold(kernel)(t, i_state, params), :chain)
    return states
end
