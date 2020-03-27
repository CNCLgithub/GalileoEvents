export extract_chain,
    to_frame

using JLD2
using DataFrames
using Gen_Compose
using Base.Filesystem
using Base.Iterators:flatten

function parse_trace(latent_maps, trace)
    d = Dict()
    for (k,f) in latent_maps
        d[k] = f(trace)
    end
    return d
end

"""
Extracts latents from an inference chain in the form of
a `Dict`.
"""
function extract_chain(path::String, latents::Dict)
    weighted = []
    unweighted = []
    log_scores = []
    jldopen(path, "r") do chain
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
function extract_chain(r::Gen_Compose.SequentialChain,
                       latents::Dict)
    extract_chain(r.path, latents)
end

function to_frame(log_scores, estimates; exclude = nothing)

    latents = keys(estimates)
    dims = size(log_scores)
    samples = collect(1:dims[1])
    columns = Dict(
        :t => repeat(samples, inner = dims[2]),
        :sid => repeat(collect(1:dims[2]), dims[1]),
        :log_score => collect(flatten(log_scores'))
    )
    for l in latents
        # (l in exclude) || columns[l] = collect(flatten(estimates[l]'))
        (l in exclude) || setindex!(columns, collect(flatten(estimates[l]')), l)

    end

    df = DataFrame(columns)
    return df
end

