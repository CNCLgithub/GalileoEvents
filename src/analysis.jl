export extract_chain,
    extract_mh_chain,
    to_frame,
    mh_to_frame,
    digest_pf_trial,
    evaluation,
    merge_evaluation

using CSV
using JLD2
using DataFrames
using DataFramesMeta
using StatsModels
using GLM
using UnicodePlots
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
        (!isnothing(exclude) && (l in exclude)) ||
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
    df = df[in.(df.t, Ref(tps)),:]
    sort!(df, :t)
end

function evaluation(dataset::String, trial::Int;
                    obs_noise::Float64 = 0.1,
                    particles::Int = 10,
                    chains::Int = 1,
                    bo_ret = false)
    d = galileo_ramp.Exp1Dataset(dataset)
    (_,_, tps) = get(d, trial)
    for i = 1:chains
        chain = seq_inference(dataset, trial, particles, obs_noise;
                              bo = true)
        # returns tibble of: | :t | :ramp_density_mean | :log_score_mean
        tibble = digest_pf_trial(chain, tps)
        tibble[!, :chain] .= i
        if i == 1
            df = tibble
        else
            df = vcat(df, tibble)
        end 
    end
    if bo_ret
        df = @linq df |>
            by(:t, rp_mean = mean(:ramp_density))
        df[!, :trial] .= trial
        df[!, :cond] = collect(0:3)
        if trial < 120
            df[!, :scene] .= Int(floor(trial/2))
            df[!, :congruent] .= (trial % 2) == 0
        else
            df[!, :scene] .= Int(trial - 60)
            df[!, :congruent] .= true
        end
    end
    df
end

function merge_evaluation(evals, responses)
    # | :scene | :congruent | :t | avg_human_response | log(v1/m2)
    human_responses = DataFrame(CSV.File(responses))
    # | :scene | :congruent | :t | :ramp_density_mean | :log_score_mean
    results = vcat(evals...)
    results = join(results, human_responses,
                   on = [:scene, :congruent, :cond])
    return results
end

"""
Computes RMSE of model predictions on human judgements.
The linear model is fit using the control trials.
"""
function fit_pf(data)
    df = @linq data |>
        where(:cond .> 0) |>
        transform(model_mass_ratio = log.(:ramp_density_mean) + :v_m2) |>
        by([:scene,:cond], model_ratio_diff = diff(:model_mass_ratio),
           human_ratio_diff = diff(:avg_human_response),
           gt_ratio_diff = diff(:gt_mass_ratio))

    heavy = lm(@formula(human_ratio_diff ~ model_ratio_diff),
               @where(df, :gt_ratio_diff .> 0))
    light = lm(@formula(human_ratio_diff ~ model_ratio_diff),
               @where(df, :gt_ratio_diff .< 0))
    plt = densityplot(df[!, :cond],
                      df[!, :human_ratio_diff],
                      name = "human")
    display(plt)
    plt = densityplot(df[!, :cond],
                      df[!, :model_ratio_diff],
                      name = "model")
    display(plt)
    # plt = scatterplot(data[!, :cond],
    #                   data[!, :human_ratio_diff],
    #                   name = "human")
    # scatterplot!(plt,
    #              data[!, :cond],
    #              predict(model),
    #              name = "predict")
    # scatterplot!(plt,
    #              data[!, :cond],
    #              data[!, :model_ratio_diff],
    #              name = "gt")
    display(heavy)
    display(light)
    println(r2(heavy))
    println(r2(light))
    return (df, heavy, light)

end
