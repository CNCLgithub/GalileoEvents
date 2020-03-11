using Gadfly
using Compose
using DataFrames
using Gen
using JLD2
using Gen_Compose
import Cairo

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

function _to_frame(log_scores, estimates)

    latents = keys(estimates)
    dims = size(log_scores)
    samples = collect(1:dims[1])
    columns = Dict(
        :t => repeat(samples, inner = dims[2]),
        :sid => repeat(collect(1:dims[2]), dims[1]),
        :log_score => collect(Base.Iterators.flatten(log_scores'))
    )
    # df[:t] = repeat(samples, inner = dims[2])
    # df[:sid] = repeat(collect(1:dims[2]), dims[1])

    for l in latents
        columns[l] = collect(Base.Iterators.flatten(estimates[l]'))
    end

    df = DataFrame(columns)
    return df
end

function plot_extract(e::Dict)
    latents = collect(keys(e["weighted"]))
    df = _to_frame(e["log_scores"], e["unweighted"])
    # first the estimates
    estimates = map(y -> Gadfly.plot(df,
                                     y = y,
                                     x = :t,
                                     color = :log_score,
                                     Gadfly.Geom.histogram2d),
                    latents)
    # last log scores
    log_scores = Gadfly.plot(df, y = :log_score, x = :t,
                             Gadfly.Geom.histogram2d)
    plot = vstack(estimates..., log_scores)
    compose(compose(context(), rectangle(), fill("white")), plot)
end
