using Gen

function test(n::Int)
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = 1.5
    cm[:initial_state => 2 => :init_pos] = 0.5
    cm[:object_physics => 1 => :friction] = 0.2
    cm[:object_physics => 2 => :friction] = 0.2
    trace, w = Gen.generate(markov_generative_model,
                            (n, default_params), cm)
    # println(get_choices(trace))
end

@time test(120);
