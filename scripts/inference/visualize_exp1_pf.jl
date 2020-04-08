using Gadfly
using Compose
import Cairo

using Gen
using Gen_Compose
using GalileoRamp
using FileIO

using ArgParse
using Base.Filesystem

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--dataset"
        help = "Exp1 Dataset"
        arg_type = String
        default = "/databases/exp1.hdf5"

        "--particles"
        help = "Number of particles"
        arg_type = Int
        default = 1

        "--obs_noise"
        help = "Observation noise"
        arg_type = Float64
        default = 0.1

        "idx"
        help = "idx of trial"
        arg_type = Int
        required = true
    end

    return parse_args(s)
end


function plot_chain(df, ml_est, latents, col_t, path)
    # first the estimates
    estimates = map(y -> Gadfly.plot(df,
                                     y = y,
                                     x = :t,
                                     xintercept = col_t,
                                     Gadfly.Geom.histogram2d(xbincount=120),
                                     Gadfly.Geom.vline,
                                     Scale.x_continuous(minvalue = 0, maxvalue = 120)),
                    latents)
    plot = vstack(estimates...)
    compose(compose(context(), rectangle(), fill("white")), plot) |>
        PNG(path);
end

function process_trial(particles::Int,
                       noise::Float64,
                       trial::Int)

    dataset_path = "/databases/exp1.hdf5"
    dataset_name = first(splitext(basename(dataset_path)))
    trace_path = "/traces/$(dataset_name)_p_$(particles)_n_$(obs_noise)/$(trial)_c_1.jld2"
    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, cols) = get(dataset, trial)

    chain_path = "$trace_path/$(trial).jld2"
    extracted = extract_chain(chain_path)

    df = to_frame(extracted["log_scores"], extracted["unweighted"],
                  exclude = [:ramp_pos])
    sort!(df, :t)
    plot_path = "$trace_path/$(trial)_plot.png"
    plot_chain(df, extracted["ml_est"], [:ramp_density], cols, plot_path)

    gt_pos = state["pos"]
    preds = extracted["unweighted"][:ramp_pos]
    viz_path = "$trace_path/$(trial)_viz.gif"
    visualize(scene, gt_pos, preds, nothing, viz_path)
    return nothing;
end
