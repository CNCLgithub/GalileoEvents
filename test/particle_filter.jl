using Revise
using GalileoEvents
using Gen
using Printf
using Plots
using GenParticleFilters
using Distributions
ENV["GKSwstype"]="160" # fixes some plotting warnings

"""
gen_trial

Generates a trial and returns the generation parameters, the true trace and the observations
"""
function gen_trial()
    # configure model paramaters
    mass_ratio = rand(Gamma(2.0, 1.0))
    obj_frictions = (rand(Uniform(0.1, 0.9)), rand(Uniform(0.1, 0.9)))
    obj_positions = (rand(Uniform(0.2, 0.8)), rand(Uniform(1.2, 1.8)))
    mprior = MaterialPrior([unknown_material])
    pprior = PhysPrior((3.0, 10.0), # mass
                    (0.5, 10.0), # friction
                    (0.2, 1.0))  # restitution
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    event_concepts = Type{<:EventRelation}[Collision]
    obs_noise = 0.05
    
    cp_params = CPParams(client, [a,b], mprior, pprior, event_concepts, obs_noise)

    t = 100

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

@gen function proposal(trace)
    # find first collision in the trace
    t = get_args(trace)[1]
    start_event_indices = [trace[:kernel=>i=>:events=>:start_event_idx] for i in 1:t]
    t1 = findfirst(x -> x == 2, start_event_indices)

    if !isnothing(t1)
        # in future, maybe gaussian rw
        trace2, delta_s, _... = Gen.regenerate(trace, select(:kernel => t1 => :events => :event))
        return trace2, delta_s
    end
    
    return trace, 0
end

"""
do_inference

Runs particle filter inference on a model and given observations
"""
function do_inference(t::Int, params::CPParams, observations::Vector{ChoiceMap}, n_particles::Int = 100, ess_thresh=0.5)
    # initialize particle filter
    state = pf_initialize(cp_model, (0, params), EmptyChoiceMap(), n_particles)

    # Then increment through each observation step
    for t in 1:length(observations)
        # Update filter state with new observation at timestep t
        pf_update!(state, (t, params), (UnknownChange(), NoChange()), observations[t])

        step_time = @elapsed begin
            # Resample and rejuvenate if the effective sample size is too low
            if effective_sample_size(state) < ess_thresh * n_particles
                # Perform residual resampling, pruning low-weight particles
                pf_resample!(state, :residual)
            end
            # Perform a rejuvenation move on past choices
            #rejuv_sel = select(:kernel => t => :events => :event)
            #pf_rejuvenate!(state, mh, (rejuv_sel,))

            
            kern(trace) = move_reweight(trace, proposal, ())
            pf_move_reweight!(state, kern)
        end

        if t % 10 == 0
            @printf "%s time steps completed (last step was %0.2f seconds)\n" t step_time
        end
    end

    # return the "unweighted" set of traces after t steps
    return get_traces(state), get_log_weights(state)
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
function plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace}, weights)
    observed_plt = plot_trace(truth, "True trajectory")
    simulated_plt = plot_trace(last(traces), "Last trace")

    (t, _) = get_args(truth)
    num_traces = length(traces)
    mass_logs = [[t[:prior => :objects => i => :mass] for t in traces] for i in 1:2]

    scores_plt = plot(1:num_traces, weights, title="Scores", xlabel="trace number", ylabel="log score")
    mass_plts = [Plots.histogram(1:num_traces, mass_logs[i], title="Mass $(i == 1 ? "Ramp object" : "Table object")", legend=false) for i in 1:2]
    ratio_plt = Plots.histogram(1:num_traces, mass_logs[1]./mass_logs[2], title="mass ramp object / mass table object", legend=false)
    plt = plot(observed_plt, simulated_plt, mass_plts..., scores_plt, ratio_plt,  size=(1200, 800))
    savefig(plt, "test/plots/particle_filter.png")
end

# data generation
t, params, truth, observations = gen_trial()
#display(get_choices(truth))

# inference
traces, weights = do_inference(t, params, observations, 100)

# visualize results
plot_traces(truth, traces, weights)