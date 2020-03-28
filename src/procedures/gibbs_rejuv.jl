export exp1_gibbs

# function extract_args(trace, objects)
#     choices = Gen.get_choices(trace)
#     [Gen.get_submap(choices, :object_physics => i) for i = objects]
# end

# @gen (static) function gibbs_chain(choices)
#     density = choices[:density]
#     friction = choices[:friction]
#     @trace(log_uniform(density, 0.1), :density)
#     # @trace(log_uniform(friction, 0.1), :friction)
# end

# map_update = Gen.Map(gibbs_chain)

# @gen (static) function gibbs_step(trace, objects)
#     args = extract_args(trace, objects)
#     @trace(map_update(args), :object_physics)
# end


@gen function exp1_gibbs(trace)
    choices = Gen.get_choices(trace)
    addr = :object_physics => 1 => :density
    density = choices[addr]
    low = density - 1.0
    high = density + 1.0
    @trace(uniform(low, high), addr)
end
