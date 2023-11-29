using Revise
using Gen
using GalileoEvents
using Plots
ENV["GKSwstype"]="160" # fixes some plotting warnings

mass_ratio = 2.0
obj_frictions = (0.3, 0.3)
obj_positions = (0.5, 1.2)

mprior = MaterialPrior([unknown_material])
pprior = PhysPrior((3.0, 10.0), # mass
                   (0.5, 10.0), # friction
                   (0.2, 1.0))  # restitution

obs_noise = 0.05
t = 120

fixed_prior_cm = Gen.choicemap()
fixed_prior_cm[:prior => :objects => 1 => :mass] = 2
fixed_prior_cm[:prior => :objects => 2 => :mass] = 1
fixed_prior_cm[:prior => :objects => 1 => :friction] = 0.5
fixed_prior_cm[:prior => :objects => 2 => :friction] = 1.2
fixed_prior_cm[:prior => :objects => 1 => :restitution] = 0.2
fixed_prior_cm[:prior => :objects => 2 => :restitution] = 0.2

function forward_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)
    
    trace, _ = Gen.generate(cp_model, (t, cp_params));
    println("")
    #display(get_choices(trace))
end

function add_rectangle!(plt, xstart, xend, y; height=0.8, color=:blue)
    xvals = [xstart, xend, xend, xstart, xstart]
    yvals = [y, y, y+height, y+height, y]
    plot!(plt, xvals, yvals, fill=true, seriestype=:shape, fillcolor=color, linecolor=color)
end

get_x2(trace, t) = get_retval(trace)[t].bullet_state.kinematics[2].position[1]

function visualize_active_events()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)    

    num_traces = 50    
    plt = plot(legend=false, xlim=(0, t), ylim=(1, num_traces+1), yrotation=90, ylabel="Trace", yticks=false, xlabel="Time step")
    collision_t = nothing
    for i in 1:num_traces
        if i % 10 == 0
            @show i
        end
        trace, _ = Gen.generate(cp_model, (t, cp_params), fixed_prior_cm);

        start = nothing
        first_x = i==1 ? get_x2(trace, 1) : nothing # only look for collision in first trace
        for j in 1:t
            if trace[:kernel=>j=>:events=>:start_event_idx]==2
                start = j
            end
            if trace[:kernel=>j=>:events=>:end_event_idx]==2
                finish = j
                add_rectangle!(plt, start, finish, i)
            end
            if first_x !== nothing && abs(first_x - get_x2(trace, j)) > 0.001
                collision_t = j
                first_x = nothing
            end
        end

    end
    vline!(plt, [collision_t], linecolor=:red, linewidth=2, label="Vertical Line")
    savefig(plt, "test/gms/plots/events.png")
end

# constrained generation, event 2 must start at timestep 10
function constrained_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    addr = 10 => :events => :start_event_idx
    cm = Gen.choicemap(addr => 2)
    trace, _ = Gen.generate(cp_model, (t, cp_params), cm)
    display(get_choices(trace))
end

# update priors
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

# change event start
function update_test_2()

    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    # generate initial trace
    trace, _ = Gen.generate(cp_model, (t, cp_params))

    # find first collision in the trace
    start_event_indices = [trace[:kernel=>i=>:events=>:start_event_idx] for i in 1:t]
    t1 = findfirst(x -> x == 2, start_event_indices)
# TODO: validate existence of event
    # move first collision five steps earlier
    cm = choicemap(fixed_prior_cm)
    cm[:kernel => t1 => :events => :start_event_idx] = 1
    cm[:kernel => t1 - 5 => :events => :start_event_idx] = 2
    trace2, ls2, _... = Gen.update(trace, cm)
    trace3, delta_s, _... = Gen.regenerate(trace2, select(:kernel => t1 - 5 => :events => :event))
    # TODO: check ls2
    # print choices and 

    @assert delta_s != -Inf
    @assert delta_s != NaN

    return trace, trace2
end

# redraw latents at same event start
function update_test_3()

    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    # generate initial trace
    trace, _ = Gen.generate(cp_model, (t, cp_params))

    # find first collision in the trace
    start_event_indices = [trace[:kernel=>i=>:events=>:start_event_idx] for i in 1:t]
    t1 = findfirst(x -> x == 2, start_event_indices)

    # in future maybe gaussian rw
    trace2, delta_s, _... = Gen.regenerate(trace, select(:kernel => t1 => :events => :event))
    
    @assert delta_s != -Inf
    @assert delta_s != NaN

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

switch = Gen.Switch(function1, function2)

@gen function switch_model_static()
    function_idx = @trace(categorical([0.5, 0.5]), :function)
    x = @trace(switch(function_idx), :x)
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

    # update and regenerate trace
    trace3, _ = Gen.update(trace, cm)
    trace4, _ = Gen.regenerate(trace3, select(:x))
    display(get_choices(trace4))
end

#forward_test()
#visualize_active_events()
#constrained_test()
#update_test()
#update_test_2()
update_test_3()
#switch_test_static()