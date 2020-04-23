using Gen

function test(n::Int)

    cid = GalileoRamp.physics.physics.init_client(direct = true)
    obj_prior = fill(GalileoRamp.default_object, 2)
    init_pos = [1.5, 0.5]
    scene = GalileoRamp.initialize_state(obj_prior, init_pos)
    params = Params(obj_prior, init_pos, scene, 0.1, cid)
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = init_pos[1]
    cm[:initial_state => 2 => :init_pos] = init_pos[2]
    cm[:object_physics => 1 => :friction] = 0.2
    cm[:object_physics => 2 => :friction] = 0.2
    trace, w = Gen.generate(cp_generative_model,
                            (n, params), cm)
    GalileoRamp.physics.physics.clear_trace(cid)
    return trace
end

test(1);
@time test(1);
@time test(120);
# trace = @time test(10);
# # println(Gen.get_choices(trace))
