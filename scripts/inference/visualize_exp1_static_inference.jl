using CSV
using PyCall

include("../../inference/visualize/visualize.jl")

np = pyimport("numpy")
gm = pyimport("galileo_ramp.world.simulation.exp2_physics");

function copy_latents!(data::Dict, df::DataFrameRow)
    data["objects"]["A"]["density"] = df[:density_a]
    data["objects"]["A"]["mass"] = data["objects"]["A"]["density"] * data["objects"]["A"]["volume"]
    data["objects"]["B"]["density"] = df[:density_b]
    data["objects"]["B"]["mass"] = data["objects"]["B"]["density"] * data["objects"]["B"]["volume"]
    return data
end

function make_predictions(scene_data, maps, dims)
    sim = Array{Float64, 4}(undef, nrow(maps), dims...)

    for (i, row) in enumerate(eachrow(maps))
        predict_data = deepcopy(scene_data)
        copy_latents!(predict_data, row)
        sim[i, :, :, :] = gm.run_full_trace(predict_data,
                                            ["A", "B"],
                                            first(dims) * 1.0/60.0,
                                            fps = 60,
                                            time_scale = 10.0)[1]
    end
    return sim
end


function gt_simulation(scene_data, dims)
    sim = Array{Float64, 4}(undef, 1, dims...)
    sim[1, :, :, :] = gm.run_full_trace(scene_data,
                                        ["A", "B"],
                                        first(dims) * 1.0/60.0,
                                        fps = 60,
                                        time_scale = 10.0)[1]
    return sim
end

function main()
    scene_dir = "../../data/galileo-ramp/scenes/legacy_converted"
    trace_dir = "../../data/galileo-ramp/traces/exp1_static_inference"
    trace_path = "../../data/galileo-ramp/traces/exp1_static_inference_summary_map.csv"
    # Output dir
    out_dir = joinpath(@__DIR__, "output")

    latents = [:density_a, :density_b]
    num_sims = 10 # number of chains

    trace_df = CSV.read(trace_path, copycols=true)

    # Loop through trials and overlay ground truth with the posterior simulation
    for trial = 0:119 # 209
        # Make one directory for each trial
        trial_out = joinpath(trace_dir, "$(trial)")

        # Load ground truth
        json_path = joinpath(scene_dir, "trial_$(trial).json")
        npy_path = joinpath(scene_dir, "trial_$(trial)_pos.npy")
        scene_data = Dict()
        open(json_path, "r") do f
            scene_data=JSON.parse(f)["scene"]
        end
        pos = np.load(npy_path)
        pos = permutedims(pos, [2, 1, 3])

        # Prepare usage of the generative model
        map_estimates = filter(r -> r[:trial_id] == trial, trace_df)
        sims = make_predictions(scene_data, map_estimates, size(pos))
        # obtain a simulation of the generative model using the gt values
        gt_sim = gt_simulation(scene_data, size(pos))

        visualize(scene_data, pos, sims, gt_sim,  trial_out)

    end
end


main()
