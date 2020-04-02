export extract_chain,
    extract_mh_chain,
    to_frame,
    mh_to_frame,
    digest_pf_chain,
    fit_pf

using JLD2
using DataFrames
using StatsModels
using Base.Iterators:flatten

function extract_mh_chain(path::String)
    estimates = []
    log_scores = []
    jldopen(path, "r") do chain
        states = chain["state"]
        for t = 1:length(keys(states))
            state = states["$t"]
            push!(estimates, state["estimates"])
            push!(log_scores, state["log_score"])
        end
    end
    estimates = merge(vcat, estimates...)
    log_scores = vcat(log_scores...)
    extracts = Dict("estimates" => estimates,
                    "log_scores" => log_scores)
    return extracts
end
function mh_to_frame(t,log_scores, estimates; exclude = nothing)

    latents = keys(estimates)
    dims = size(log_scores)
    samples = collect(1:dims[1])
    println(size(estimates[:ramp_density]))
    columns = Dict(
        :t => fill(t, dims[1]),
        # :sid => repeat(collect(1:dims[2]), inner = dims[1]),
        :log_score => collect(flatten(log_scores))
    )
    for l in latents
        (l in exclude) ||
            setindex!(columns, collect(flatten(estimates[l])), l)

    end
    df = DataFrame(columns)
    return df
end

"""
Extracts latents from an inference chain in the form of
a `Dict`.
"""
function extract_chain(path::String)
    weighted = []
    unweighted = []
    log_scores = []
    ml_est = []
    states = []
    jldopen(path, "r") do chain
        states = chain["state"]
        for t = 1:length(keys(states))
            state = states["$t"]
            push!(weighted, state["weighted"])
            push!(unweighted, state["unweighted"])
            push!(log_scores, state["log_scores"])
            push!(ml_est, state["ml_est"])
        end
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = vcat(log_scores...)
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores,
                    "ml_est" => ml_est)
    return extracts
end
function extract_chain(r::Gen_Compose.SequentialChain)
    weighted = []
    unweighted = []
    log_scores = []
    ml_est = []
    states = []
    for t = 1:length(r.buffer)
        state = r.buffer[t]
        push!(weighted, state["weighted"])
        push!(unweighted, state["unweighted"])
        push!(log_scores, state["log_scores"])
        push!(ml_est, state["ml_est"])
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = vcat(log_scores...)
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores,
                    "ml_est" => ml_est)
    return extracts
end

function to_frame(log_scores, estimates; exclude = nothing)

    latents = keys(estimates)
    dims = size(log_scores)
    samples = collect(1:dims[1])
    columns = Dict(
        :t => repeat(samples, dims[2]),
        :sid => repeat(collect(1:dims[2]), inner = dims[1]),
        :log_score => collect(flatten(log_scores))
    )
    for l in latents
        (l in exclude) ||
            setindex!(columns, collect(flatten(vcat(estimates[l]...))), l)

    end

    df = DataFrame(columns)
    return df
end

"""
Returns a tibble of average model estimates for each time point.
"""
function digest_pf_trial(chain, tps)
    extracted = extract_chain(chain)
    df = to_frame(extracted["log_scores"], extracted["unweighted"])
    df = df[in.(df.t, Ref(cols)),:]
    sort!(df, :t)
    aggregate(groupby(df, :t), mean)
end

"""
Computes RMSE of model predictions on human judgements.
The linear model is fit using the control trials.
"""
function fit_pf(data)
    model = fit(LinearModel,
                @formula(avg_human_response ~ avg_model_estimates),
                filter(row -> row[:scene] >= 60, data))
    preds = predict(model, filter(row -> row[:scene] < 60, data))
    resids = residuals(preds)
    rmse = sqrt((1.0/length(resids) * rss(redis))
end
