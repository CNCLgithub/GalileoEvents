export gibbs_step

function extract_args(trace)
    (t, p) = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    [Gen.get_submap(choices, :object_physics => i) for i = 1:p.n_objects]
end

@gen (static) function gibbs_chain(choices)
    density = choices[:density]
    friction = choices[:friction]
    @trace(log_uniform(density, 0.1), :density)
    @trace(log_uniform(friction, 0.1), :friction)
end

map_update = Gen.Map(gibbs_chain)

@gen (static) function gibbs_step(trace)
    args = extract_args(trace)
    @trace(map_update(args), :object_physics)
end
