using Gadfly
using Compose
import Cairo

using Gen
using Gen_Compose
using GalileoRamp
using FileIO



function plot_chain(df, latents, col_t, path)
    # first the estimates
    estimates = map(l -> Gadfly.plot(df,
                                     x = l,
                                     # xintercept = col_t,
                                     Gadfly.Geom.histogram),
                    latents)
    # last log scores
    # log_scores = Gadfly.plot(x = 1:120,
    #                          y = ml_est,
    #                          xintercept = [col_t],
    #                          Gadfly.Geom.point,
    #                          Gadfly.Geom.vline,
    #                          Scale.x_continuous(minvalue = 0, maxvalue = 120))
    # log_scores = Gadfly.plot(df, y = :log_score, x = :t, xintercept = [col_t],
    #                          Gadfly.Geom.histogram2d(ybincount=20),
    #                          Gadfly.Geom.vline,
    #                          Scale.x_continuous(minvalue = 0, maxvalue = 120))
    # plot = vstack(estimates..., log_scores)
    plot = vstack(estimates...)
    compose(compose(context(), rectangle(), fill("white")), plot) |>
        PNG(path);
end

function process_trial(dataset_path,
                       trace_path::String,
                       trial::Int)

    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, cols) = get(dataset, trial)

    chain_path = "$trace_path/$trial.jld2"
    extracted = extract_mh_chain(chain_path)

    df = mh_to_frame(120, extracted["log_scores"], extracted["estimates"],
                  exclude = [:ramp_pos])
    sort!(df, :t)
    plot_path = "$trace_path/$(trial)_plot.png"
    plot_chain(df, [:ramp_density], cols, plot_path)

    n = 100
    sorted = reverse(sortperm(extracted["log_scores"]))[1:n]
    println(extracted["log_scores"][sorted])
    preds = extracted["estimates"][:ramp_pos][sorted, :, :, :]
    println(size(preds))
    # println(size(permutedims(preds, (1, 0, 2, 3))))
    pred_pos = []
    for t = 1:120
        push!(pred_pos, reshape(preds[:,t,:,:], (1,n,2,3)))
    end
    pred_pos = vcat(pred_pos...)
    println(size(pred_pos))
    gt_pos = state["pos"]
    # preds = extracted["unweighted"][:ramp_pos]
    viz_path = "$trace_path/$(trial)_viz.gif"
    visualize(scene, gt_pos, pred_pos, nothing, viz_path)
    return nothing;
end

process_trial("/databases/exp1.hdf5", "/traces", 0);
