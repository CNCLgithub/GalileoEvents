using Revise
using GalileoEvents
using Gen
using Printf
using Plots
ENV["GKSwstype"]="160" # fixes some plotting warnings

"""
gen_trial

Generates a trial and returns the generation parameters, the true trace and the observations
"""
function gen_trial()
    # configure model paramaters
    mass_ratio = 2.0
    obj_frictions = (0.3, 0.3)
    obj_positions = (0.5, 1.2)
    mprior = MaterialPrior([unknown_material])
    pprior = PhysPrior((3.0, 10.0), # mass
                    (0.5, 10.0), # friction
                    (0.2, 1.0))  # restitution
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    obs_noise = 0.05
    
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    t = 60

    # run model forward
    trace, _ = Gen.generate(cp_model, (t, cp_params));

    # collect observations
    choices = get_choices(trace)
    observations = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observe
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        observations[i] = cm
    end

    return t, cp_params, trace, observations
end

"""
do_inference

Implements a truncated random walk for the both mass priors
"""
@gen function proposal(tr::Gen.Trace)
    # get previous values from `tr`
    address(obj_nr) = :prior => :objects => obj_nr => :mass
    choices = get_choices(tr)
    #display(choices)
    for i in 1:2
        prev_mass = choices[address(1)]
        mass = {address(i)} ~ trunc_norm(prev_mass, .25, 0., Inf)
    end
end

"""
do_inference

Runs particle filter inference on a model and given observations
"""
function do_inference(t::Int, params::CPParams, observations::Vector{ChoiceMap}, n_particles::Int = 100)
    # initialize particle filter
    state = Gen.initialize_particle_filter(cp_model, (0, params), EmptyChoiceMap(), n_particles)

    # Then increment through each observation step
    for (t, o) = enumerate(observations)
        # apply a rejuvenation move to each particle
        step_time = @elapsed begin
            for i=1:particles
                state.traces[i], _  = mh(state.traces[i], proposal, ())
            end
        
            Gen.maybe_resample!(state, ess_threshold=particles/2) 
            Gen.particle_filter_step!(state, (t, params), (UnknownChange(), NoChange()), o)
        end

        if t % 10 == 0
            @printf "%s time steps completed (last step was %0.2f seconds)\n" t step_time
        end
    end

    # return the "unweighted" set of traces after t steps
    return Gen.sample_unweighted_traces(state, particles)
end


function plot_trace(tr::Gen.Trace, title="Trajectory")
    (t, _) = get_args(tr)
    # get the prior choice for the two masses
    choices = get_choices(tr)
    masses = [round(choices[:prior => :objects => i => :mass], digits=2) for i in 1:2]

    # get the x positions
    states = get_retval(tr)
    #diplsay(states)
    xs = [map(st -> st.bullet_state.kinematics[i].position[1], states) for i = 1:2]

    # return plot
    plot(1:t, xs, title=title, labels=["ramp: $(masses[1])" "table: $(masses[2])"], xlabel="t", ylabel="x")
end

"""
plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})

Display the observed and final simulated trajectory as well as distributions for latents and the score
"""
function plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})
    observed_plt = plot_trace(truth, "True trajectory")
    simulated_plt = plot_trace(last(traces), "Last trace")

    (t, _) = get_args(truth)
    num_traces = length(traces)
    mass_logs = [[t[:prior => :objects => i => :mass] for t in traces] for i in 1:2]
    scores = [get_score(t) for t in traces]

    scores_plt = plot(1:num_traces, scores, title="Scores", xlabel="trace number", ylabel="log score")
    mass_plts = [Plots.histogram(1:num_traces, mass_logs[i], title="Mass $(i == 1 ? "Ramp object" : "Table object")", legend=false) for i in 1:2]
    ratio_plt = Plots.histogram(1:num_traces, mass_logs[1]./mass_logs[2], title="mass ramp object / mass table object", legend=false)
    plt = plot(observed_plt, simulated_plt, mass_plts..., scores_plt, ratio_plt,  size=(1200, 800))
    savefig(plt, "test/plots/particle_filter.png")
end

# data generation
t, params, truth, observations = gen_trial()
#display(get_choices(truth))

# inference
traces = do_inference(t, params, observations, 5)

# visualize results
plot_traces(truth, traces)