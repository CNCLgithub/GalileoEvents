export exp1_mc_gibbs,
    exp1_mx_density

@gen function exp1_mc_gibbs(trace)
    choices = Gen.get_choices(trace)
    addr = :object_physics => 1 => :density
    density = choices[addr]
    low = density - 1.0
    high = density + 1.0
    @trace(uniform(low, high), addr)
end

@gen function exp1_mx_density(trace)
    i,params = get_args(trace)
    addr = :chain => i => :physics => 1 => :density
    choices = Gen.get_choices(trace)
    d = choices[addr]
    @trace(log_uniform(d, 0.2), addr)
end
