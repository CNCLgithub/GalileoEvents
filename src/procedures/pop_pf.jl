export PopParticleFilter


struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    proposal::Union{Gen.GenerativeFunction, Nothing}
    prop_args::Tuple
    rejuvination::Union{Function, Nothing}
    pop_stats::Union{Function, Nothing}
    stop_rejuv::Union{Function, Nothing}
    max_sweeps::Int
    max_count::Int
    verbose::Bool
end

mutable struct RejuvTrace
    attempts::Int
    acceptance::Float64
    stats::Any
end

function Gen_Compose.rejuvinate!(proc::PopParticleFilter,
                                 state::Gen.ParticleFilterState)
    rtrace = RejuvTrace(0, 0, nothing)
    rtrace.stats = proc.pop_stats(state)

    t, _ = Gen.get_args(first(state.traces))

    if isnothing(proc.rejuvination) || proc.stop_rejuv(rtrace.stats)
        return rtrace
    end

    fails = 0
    for sweep = 1:proc.max_sweeps
        rtrace.acceptance += proc.rejuvination(state, rtrace.stats)
        rtrace.attempts += 1
        new_stats = proc.pop_stats(state)

        if proc.stop_rejuv(new_stats, rtrace.stats)
            fails += 1
            (fails == proc.max_count) && break
        else
            fails = 0
        end

        rtrace.stats = new_stats
    end
    rtrace.acceptance = rtrace.acceptance / rtrace.attempts
    return rtrace
end

function Gen_Compose.initialize_procedure(proc::ParticleFilter,
                                          query::StaticQuery)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           query.args,
                                           query.observations,
                                           proc.particles)
    rejuvinate!(proc, state)
    return state
end

function Gen_Compose.smc_step!(state::Gen.ParticleFilterState,
                               proc::PopParticleFilter,
                               query::StaticQuery)
    # Resample before moving on...
    # TODO: Potentially bad for initial step
    Gen_Compose.resample!(proc, state, true)
    Gen.maybe_resample!(state, ess_threshold=proc.ess, verbose=true)

    # update the state of the particles
    if isnothing(proc.proposal)
        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations)
    else
        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations,
                                  proc.proposal,
                                  (query.observations, proc.prop_args...))
    end

    aux_contex = nothing

    if !isnothing(proc.rejuvination)
        aux_contex = Gen_Compose.rejuvinate!(proc, state)
    end

    return aux_contex
end
