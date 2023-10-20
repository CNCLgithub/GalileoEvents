using Gen
using GalileoEvents

mass_ratio = 2.0
obj_frictions = (0.3, 0.3)
obj_positions = (0.5, 1.2)

mprior = MaterialPrior([unknown_material])
pprior = PhysPrior((3.0, 10.0), # mass
                   (0.5, 10.0), # friction
                   (0.2, 1.0))  # restitution

obs_noise = 0.05
t = 120

function forward_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)
    addr = :prior => :objects => 1 => :mass
    cm = Gen.choicemap(addr => 30)
    trace, _ = Gen.generate(cp_model, (t, cp_params), cm)
    #display(get_choices(trace))
end

# constrained generation
function constrained_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    addr = 10 => :events => :start_event_idx
    cm = Gen.choicemap(addr => 1)
    trace, _ = Gen.generate(cp_model, (t, cp_params), cm)
    display(get_choices(trace))
end

# gen regenerate


function update_test()
    t = 120

    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)
    trace, _ = Gen.generate(cp_model, (t, cp_params))

    addr = :prior => :objects => 1 => :mass
    cm = Gen.choicemap(addr => trace[addr] + 3)
    trace2, _ = Gen.update(trace, cm)

    # compare final positions
    t=120
    pos1 = Vector(get_retval(trace)[t].bullet_state.kinematics[1].position)
    pos2 = Vector(get_retval(trace2)[t].bullet_state.kinematics[1].position)
    @assert pos1 != pos2

    return trace, trace2
end


function update_test_2()
    t = 120

    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)
    trace, _ = Gen.generate(cp_model, (t, cp_params))

    addr = :prior => :objects => 1 => :mass
    cm = Gen.choicemap(addr => trace[addr] + 3)
    trace2, _ = Gen.update(trace, cm)

    # compare final positions
    t=120
    pos1 = Vector(get_retval(trace)[t].bullet_state.kinematics[1].position)
    pos2 = Vector(get_retval(trace2)[t].bullet_state.kinematics[1].position)
    @assert pos1 != pos2

    # TODO: more tests that include events

    return trace, trace2
end


# test switch combinator in terms of gen's reaction to proposed changes

# toy model for dealing with complexing
# random walk with 2 delta functions (gaussian vs uniform) chosen by switch
# initial trace is changed by a mh proposal for switch index
# static first, unfold complexity second step

@gen function function1() 
    v ~ normal(0., 1.)
end

@gen function function2()
    v ~ uniform(-1., 1.)
end

@gen function switch_model_static()
    function_idx = @trace(categorical([0.5, 0.5]), :function)

    x = @trace(Gen.Switch(function1, function2)(function_idx), :x)
    
    y = @trace(normal(x, 1.), :y)
end

function switch_test_static()
    # unconstrained generation
    trace, _ = Gen.generate(switch_model_static, ())
    display(get_choices(trace))

    # constrained generation
    cm = Gen.choicemap(:function => 1)
    trace2, _ = Gen.generate(switch_model_static, (), cm)
    display(get_choices(trace2))

    # update trace
    trace3, _ = Gen.update(trace, cm)
    display(get_choices(trace3))
end



@gen function switch_model_unfold()
    function_idx = @trace(categorical([0.5, 0.5]), :function)

    x = @trace(Gen.Switch(function1, function2)(function_idx), :x)
    
    y = @trace(normal(x, 1.), :y)
end


switch_test_static()
#forward_test()
#constrained_test()
#update_test()