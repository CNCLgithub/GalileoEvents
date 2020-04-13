
export mixture_generative_model

## Generative Model + components

const material_ps = ones(3) ./ 3
const incongruent_mat = Dict( "density" => (4.0, 20.0),
                              "lateralFriction" => (0.3, 0.5))


@gen (static) function object_prior(material_params)
    material = @trace(categorical(material_ps), :material)
    from_mat = from_material_params(material)
    congruent = @trace(bernoulli(0.9), :congruent)
    prior = congruent ? from_mat : incongruent_mat
    density = prior["density"]
    friction = prior["lateralFriction"]
    dens = @trace(trunc_norm(density[1], density[2], 0., 150.),
                  :density)
    fric = @trace(trunc_norm(friction[1], friction[2], 0., 1.),
                  :friction)
    restitution = @trace(uniform(0.8, 1.0), :restitution)
    physical_props = Dict("density" => dens,
                          "lateralFriction" => fric,
                          "restitution" => restitution,
                          "congruent" => congruent)
    return physical_props
end

map_object_prior = Gen.Map(object_prior)

@gen (static) function state_prior()
    init_pos = @trace(uniform(0, 2), :init_pos)
    return init_pos
end

map_init_state = Gen.Map(state_prior)

function _helper(prev_phys, switch, mat)
    prev_con = Bool(prev_phys["congruent"])
    if switch
        prior = prev_con ? incongruent_mat : from_material_params(mat)
    else
        prior = Dict( "density" => (prev_phys["density"], 0.1))
    end
    return prior
end

@gen (static) function object_kernel(prev_phys::Dict{String, Float64},
                                     mat_info::Dict)
    prev_con = Bool(prev_phys["congruent"])
    switch_p = prev_con ? 0.01 : 0.001
    switch = @trace(bernoulli(switch_p), :switch)
    prior = _helper(prev_phys, switch, mat_info)
    density = prior["density"]
    dens = @trace(trunc_norm(density[1], density[2], 0., 150.),
                  :density)
    # cong switch
    # t f t
    # t t f
    # f t t
    # f f f
    con = prev_con ⊻ switch
    physical_props = Dict{String, Float64}("density" => dens,
                          "lateralFriction" => prev_phys["lateralFriction"],
                          "restitution" => prev_phys["restitution"],
                          "congruent" => con)
    return physical_props
end

map_obj_kernel = Gen.Map(object_kernel)

@gen (static) function kernel(t::Int, prev::Tuple, params::Params)
    # prev_state, prev_phys = prev
    belief = @trace(map_obj_kernel(prev[2], params.object_prior), :physics)
    next_state = forward_step(prev[1], params, belief)
    pos = next_state[1, :, :]
    next_pos = @trace(Gen.broadcasted_normal(pos, params.obs_noise),
                      :positions)
    nxt = (next_state, belief)
    return nxt
end

chain = Gen.Unfold(kernel)

@gen (static) function mixture_generative_model(t::Int, params::Params)

    objects = @trace(map_object_prior(params.object_prior),
                     :object_physics)
    init_args = fill(tuple(), params.n_objects)
    initial_pos = @trace(map_init_state(init_args), :initial_state)
    i_state = initialize_state(params, objects, initial_pos)

    states = @trace(chain(t, i_state, params), :chain)
    return states
end
