using Gen
using Gen_Compose

struct BatchInference <: Gen_Compose.InferenceProcedure
    update::T where T<:Function
end

mutable struct BatchInferenceTrace
    current_trace::T where T<:Gen.DynamicDSLTrace
end

function Gen_Compose.initialize_procedure(proc::BatchInference,
                              query::StaticQuery,
                              addr)
    addr = observation_address(query)
    trace,_ = Gen.generate(query.forward_function,
                           (query.prior, query.args..., addr),
                           query.observations)
    return BatchInferenceTrace(trace)
end

function Gen_Compose.step_procedure!(state::BatchInferenceTrace,
                         proc::BatchInference,
                         query::StaticQuery,
                         addr,
                         step_func)
    state.current_trace = proc.update(state.current_trace)
    return nothing
end

function Gen_Compose.report_step!(results::T where T<:Gen_Compose.InferenceResult,
                      state::BatchInferenceTrace,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    trace = state.current_trace
    results.log_score[idx] = Gen.get_score(trace)
    choices = Gen.get_choices(trace)
    for l = 1:length(latents)
        results.estimates[idx,1, l] = choices[latents[l]]
    end
    return nothing
end

function Gen_Compose.initialize_results(::BatchInference)
    (1,)
end
