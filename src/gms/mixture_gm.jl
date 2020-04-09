
export mixture_generative_model

## Generative Model + components

@gen (static) function object_physics(density, friction)
    density = @trace(trunc_norm(density[1], density[2],
                                0., 150.),
                     :density)
    friction = @trace(trunc_norm(friction[1], friction[2],
                                 0., 1.),
                      :friction)
    restitution = @trace(uniform(0.8, 1.0), :restitution)
    physical_props = Dict("density" => density,
                          "lateralFriction" => friction,
                          "restitution" => restitution)
    return physical_props
end

const incongruent_mat = Dict( "density" => (4.0, 20.0),
                              "friction" => (0.3, 0.5))

@gen (static) function object_prior(material_params)
    from_mat = from_material_params(material_params)
    congruent = @trace(binomial(0.9), :congruent)
    prior = congruent ? from_mat : incongruent_mat
    density = prior["density"]
    friction = prior["friction"]
    properties = @trace(object_physics(density, friction), :physics)
    properties["congruent"] = congruent
    return properties
end

map_object_prior = Gen.Map(object_prior)

@gen (static) function state_prior()
    init_pos = @trace(uniform(0, 2), :init_pos)
    return init_pos
end

map_init_state = Gen.Map(state_prior)

@gen (static) function object_kernel(prev_phys::Dict{String, Float64})
    switch = @trace(binomial(0.1), :switch)
    prior = _helper(prev_phys, switch)
    density = prior["density"]
    friction = prior["friction"]
    properties = @trace(object_physics(density, friction), :physics)
    # cong switch
    # t f t
    # t t f
    # f t t
    # f f f
    properties["congruent"] = prev_phys["congruent"] xor switch
    return properties
end

map_obj_kernel = Gen.Map(object_kernel)

@gen (static) function kernel(t::Int, prev::Tuple, params::Params)
    prev_phys, prev_state = prev
    belief = @trace(map_obj_kernel(prev_phys), :property_kernel)
    next_state = forward_step(prev_state, params, belief)
    pos = next_state[1, :, :]
    next_pos = @trace(Gen.broadcasted_normal(pos, params.obs_noise),
                      :positions)
    nxt = (belief, next_state)
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
