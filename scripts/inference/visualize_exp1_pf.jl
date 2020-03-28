using Gadfly
using Compose
import Cairo

using Gen
using Gen_Compose
using GalileoRamp
using FileIO

function extract_pos(t)
    ret = Gen.get_retval(t)[end]
    pos = ret[1, :, :]
    pos = reshape(pos, (1, 1, size(pos)...))
    return pos
end

function extract_phys(t, feat)
    d = Vector{Float64}(undef, 1)
    d[1] = Gen.get_choices(t)[:object_physics => 1 => feat]
    reshape(d, (1,1,1))
end

extract_map = Dict(
    :pos => extract_pos,
    :density => t -> extract_phys(t, :density),
    # :friction => t -> extract_phys(t, :friction),
)

function plot_chain(df, latents, path)
    # first the estimates
    estimates = map(y -> Gadfly.plot(df,
                                     y = y,
                                     x = :t,
                                     Gadfly.Geom.histogram2d),
                    latents)
    # last log scores
    log_scores = Gadfly.plot(df, y = :log_score, x = :t,
                             Gadfly.Geom.histogram2d)
    plot = vstack(estimates..., log_scores)
    compose(compose(context(), rectangle(), fill("white")), plot) |>
        PNG(path);
end

function process_trial(dataset_path,
                       trace_path::String,
                       trial::Int)


    chain_path = "$trace_path/$trial.jld2"
    extracted = extract_chain(chain_path, extract_map)
    println(size(extracted["weighted"][:density]))

    df = to_frame(extracted["log_scores"], extracted["unweighted"],
                  exclude = [:pos])
    sort!(df, :t)
    plot_path = "$trace_path/$(trial)_plot.png"
    plot_chain(df, [:density], plot_path)

    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, _) = get(dataset, trial)
    gt_pos = state["pos"]
    preds = extracted["unweighted"][:pos]
    viz_path = "$trace_path/$(trial)_viz.gif"
    visualize(scene, gt_pos, preds, nothing, viz_path)
    return nothing;
end

process_trial("/databases/exp1.hdf5", "/traces/exp1_p_10", 0);
