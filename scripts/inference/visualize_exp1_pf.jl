using Gadfly
using Compose
import Cairo

using Gen
using Gen_Compose
using GalileoRamp
using FileIO

using ArgParse
using Base.Filesystem

using Glob

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


function plot_chain(df, col_t, path)

    density = Gadfly.plot(df,
                          x = :t,
                          y = :ramp_density,
                          Gadfly.Geom.histogram2d(xbincount=120,
                                                  ybincount=20),
                          Scale.y_log(),
                          xintercept = col_t,
                          Gadfly.Geom.vline,
                          Scale.x_continuous(minvalue = 0, maxvalue = 120))
    congruent = Gadfly.plot(df,
                            x = :t,
                            y = :ramp_congruent,
                            Gadfly.Geom.histogram2d(ybincount = 2,
                                                    xbincount=120),
                            xintercept = col_t,
                            Gadfly.Geom.vline,
                            Scale.x_continuous(minvalue = 0, maxvalue = 120))
    plot = vstack(density, congruent)
    # log_scores |> PNG(path);
    compose(compose(context(), rectangle(), fill("white")), plot) |>
        PNG(path);
end

function process_trial(particles::Int,
                       obs_noise::Float64,
                       trial::Int)

    dataset_path = "/databases/exp1.hdf5"
    dataset_name = first(splitext(basename(dataset_path)))
    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, cols) = get(dataset, trial)

    trace_path = "/traces/$(dataset_name)_p_$(particles)_n_$(obs_noise)"
    chain_paths = glob("$(trial)_c_*.jld2", "$(trace_path)")
    println(chain_paths)
    if isempty(chain_paths)
        extracted = extract_chain("$(trace_path)/$(trial).jld2")
    else
        extracts = map(extract_chain, chain_paths)
        densities = hcat(map(e -> e["unweighted"][:ramp_density], extracts)...)
        positions = hcat(map(e -> e["unweighted"][:ramp_pos], extracts)...)
        extracted = Dict("log_scores" => merge(hcat, extracts...)["log_scores"],
                    "unweighted" => Dict(:ramp_density => densities,
                                        :ramp_pos => positions))
    end

    df = to_frame(extracted["log_scores"], extracted["unweighted"],
                  exclude = [:ramp_pos])
    sort!(df, :t)
    plot_path = "$trace_path/$(trial)_plot.png"
    plot_chain(df, cols, plot_path)

    gt_pos = state["pos"]
    preds = extracted["unweighted"][:ramp_pos]
    viz_path = "$trace_path/$(trial)_viz.gif"
    println(gt_pos[end,1,:])
    println(preds[end,1,1,:])
    visualize(scene, gt_pos, preds, nothing, viz_path)
    return nothing;
end
