export PopParticleFilter


struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    proposal::Union{Gen.GenerativeFunction, Nothing}
    prop_args::Tuple
    rejuvination::Union{Function, Nothing}
    verbose::Bool
end

mutable struct RejuvTrace
    attempts::Int
    acceptance::Float64
    stats::Any
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
    Gen_Compose.resample!(proc, state)
    Gen.maybe_resample!(state, ess_threshold=proc.ess)

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

    if isnothing(proc.rejuvination)
        aux_contex = nothing
    else
        aux_contex = proc.rejuvination(proc, state)
    end

    return aux_contex
end
