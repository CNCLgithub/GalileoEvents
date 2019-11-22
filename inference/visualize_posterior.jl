

function main()
    # Output dir
    out_dir = joinpath(@__DIR__, "output")
    make_path(out_dir)

    latents = [:mass_a, :mass_b, :friction_a, :friction_b, :friction_ground]
    num_sims = 1

    # Loop through trials and overlay ground truth with the posterior simulation
    for trial = [24, 88] # 0:209  # 209
        # Make one directory for each trial
        trial_out_dir = joinpath(out_dir, @sprintf("trial_%03d", trial))
        make_path(trial_out_dir)

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
        map_estimate = DataFrame(map_estimate)
        scene = Scene(scene_data, first(size(pos)), gm.run_full_trace, 0.0001)
        mass_rv = StaticDistribution{Float64}(uniform, (0.1, 200))
        friction_rv = StaticDistribution{Float64}(uniform, (0.001, 0.999))
        prior = Gen_Compose.DeferredPrior(latents,
                                          [mass_rv,
                                           mass_rv,
                                           friction_rv,
                                           friction_rv,
                                           friction_rv])

        # simulate predictions
        params = [prior, scene, :pos]
        predictions = Array{Float64, 4}(undef, 1, size(pos)...)
        make_predictions!(predictions, generative_model, params,
                                               map_estimate)

        # mov = Movie(750, 300, "$(trial)", 1:first(size(pos)))
        # backdrop(scene, framenumber) = background("white")
        # function frame(scene, framenumber)
        #     origin()
        #     translate(Point(0.,100.))
        #     # background("white")
        #     draw_groundtruth(scene_data, pos[framenumber, :, :])
        #     draw_simulations(predictions[:, framenumber, :, :])
        # end
        # animate(mov,
        #         [Luxor.Scene(mov, backdrop, 1:first(size(pos))),
        #          Luxor.Scene(mov, frame, 1:first(size(pos)),
        #                      easingfunction=easeinoutcubic)],
        #         creategif=true,
        #         pathname="$(trial_out_dir).gif"
        #         )
        for t = 1:first(size(pos))
            # Initialize drawing
            Drawing(750, 300, joinpath(trial_out_dir, "time_$(t).png"))
            origin()
            translate(Point(0.,100.))
            background("white")
            draw_groundtruth(scene_data, pos[t, :, :])
            draw_simulations(predictions[:, t, :, :])

            finish()
        end
        println("Done with trial $trial.")
    end
end


main()
