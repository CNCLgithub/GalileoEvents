using CSV
using PyCall

include("../../inference/visualize/visualize.jl")

np = pyimport("numpy")
gm = pyimport("galileo_ramp.world.simulation.exp2_physics");

function copy_latents!(data::Dict, df::DataFrameRow)
    for latent in names(df)
        data[String(latent)] = df[latent]
    end
end

function main()
    scene_dir = "../data/galileo-ramp/scenes/legacy_converted"
    trace_dir = "../data/galileo-ramp/traces/match_legacy_mh_mass"
    # Output dir
    out_dir = joinpath(@__DIR__, "output")

    latents = [:density_a, :density_b]
    num_sims = 1

    # Loop through trials and overlay ground truth with the posterior simulation
    for trial = 0:209  # 209
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
        trace_df_path = joinpath(trace_dir, "trial_$(trial)_trace.csv")
        trace_df = CSV.read(trace_df_path, copycols=true)
        map_estimate = trace_df[argmax(trace_df[:, :log_score]), latents]
        predict_data = deepcopy(scene_data)
        copy_latents!(predict_data, map_estimate)
        sim = Array{Float64, 4}(undef, 1, size(pos)...)
        sim[1, :, :, :] = gm.run_full_trace(predict_data,
                                            ["A", "B"],
                                            first(size(pos)) * 1.0/60.0,
                                            fps = 60,
                                            time_scale = 10.0)[1]
        visualize(scene_data, pos, sim, trial_out)

    end
end


main()
