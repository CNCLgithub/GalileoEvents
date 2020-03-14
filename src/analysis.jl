export extract_chain,
    to_frame

using JLD2
using DataFrames
using Gen_Compose
using Base.Filesystem

function parse_trace(latent_maps, trace)
    choices = Gen.get_choices(trace)
    d = Dict()
    for (k,f) in latent_maps
        d[k] = f(choices)
    end
    return d
end

"""
Extracts latents from an inference chain in the form of
a `Dict`.
"""
function extract_chain(r::Gen_Compose.SequentialTraceResult,
                       latents::Dict)


    weighted = []
    unweighted = []
    log_scores = []
    jldopen(r.path, "r") do chain
        states = chain["state"]
        for t = 1:length(keys(states))
            state = states["$t"]
            n = length(state.traces)
            w_traces = Gen.get_traces(state)
            uw_traces = Gen.sample_unweighted_traces(state, n)

            parsed = map(t -> parse_trace(latents, t), w_traces)
            parsed = merge(hcat, parsed...)
            push!(weighted, parsed)
            parsed = map(t -> parse_trace(latents, t), uw_traces)
            parsed = merge(hcat, parsed...)
            push!(unweighted, parsed)
            push!(log_scores, get_log_weights(state))
        end
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = hcat(log_scores...)'
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores)
    return extracts
end

function to_frame(log_scores, estimates)

    latents = keys(estimates)
    dims = size(log_scores)
    samples = collect(1:dims[1])
    columns = Dict(
        :t => repeat(samples, inner = dims[2]),
        :sid => repeat(collect(1:dims[2]), dims[1]),
        :log_score => collect(Base.Iterators.flatten(log_scores'))
    )
    for l in latents
        columns[l] = collect(Base.Iterators.flatten(estimates[l]'))
    end

    df = DataFrame(columns)
    return df
end

