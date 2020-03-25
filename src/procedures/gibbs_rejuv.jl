export gibbs_step

@gen function gibbs_step(trace)
    (t, p) = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    for i = 1:p.n_objects
        density = choices[:object_physics => i => :density]
        friction = choices[:object_physics => i => :friction]
        @trace(log_uniform(density, 0.1), :object_physics => i => :density)
        @trace(log_uniform(friction, 0.1), :object_physics => i => :friction)
    end
end
