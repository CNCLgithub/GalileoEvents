using Luxor
using PyCall
using JSON
using Printf
using Gen
using DataFrames
include("./queries/match_legacy_physics.jl")

np = pyimport("numpy")

#### DEFINE CONSTANTS

# Define data directories
scene_dir = "/gpfs/milgram/project/yildirim/mario/data/galileo-ramp/scenes/legacy_converted"  # ground truth
trace_dir = "/gpfs/milgram/project/yildirim/mario/data/galileo-ramp/traces/match_legacy"  # inference traces

# number of best-scoring SIDs to use for the forward simulations
num_sids = 15

# Define color map for different materials
color_dict = Dict{String,String}("Iron" => "silver",
                                 "Wood" => "burlywood",
                                 "Brick" => "firebrick",
                                 "ramp" => "black",
                                 "table" => "black")

# global scaling
scale_fac = -10

########################


function draw_box(obj_data, pos, orientation)
    pos = pos*scale_fac
    dim = obj_data["dims"]*scale_fac
    sethue(color_dict[obj_data["appearance"]])
    gsave()
    translate(pos)
    rotate(-orientation)
    box(O, dim[1], dim[3], :stroke)
    grestore()
end

function draw_groundtruth(scene_data, obj_positions)
    data = deepcopy(scene_data)
    # loop through objects
    for item in keys(data)
        if item != "objects"
            obj = data[item]
            pos = Point(obj["position"][1], obj["position"][3])
            orientation = obj["orientation"][2]
            draw_box(obj, pos, orientation)
        else
            for (idx, sub_item) in enumerate(sort(collect(keys(data[item]))))
                obj = data[item][sub_item]
                pos = Point(obj_positions[idx, 1], obj_positions[idx, 3])
                draw_box(obj, pos, 0)
            end
        end
    end
end

function make_path(path)
    if !isdir(path)
        mkpath(path)
    end
end

function simulate_latents(forward_model, curr_state, params, trace_df::DataFrame)
    t = params[1]  # time step
    num_sids = first(size(curr_state))

    # empty vector to store observations for selected particles and this time step
    new_state = Vector{Any}(undef, num_sids)
    new_pos = Vector{Array{Float64, 2}}(undef, num_sids)

    for sid = 1:num_sids
        params_cp = deepcopy(params)
        trace = trace_df[[sid], :]
        choices = choicemap((:friction_a, trace[!, :friction_a][1]),
                            (:friction_b, trace[!, :friction_b][1]),
                            (:friction_ground, trace[!, :friction_ground][1]))

        prepend!(params_cp, [curr_state[sid]])

        (trace, _) = generate(forward_model, Tuple(params_cp), choices)
        new_state[sid] = get_retval(trace)
        new_pos[sid] = get_choices(trace)[params_cp[5]]
    end
    return (new_state, new_pos)
end

function draw_simulation(positions)
    num_sids = first(size(positions))
    for sid = 1:num_sids
        pos = positions[sid]
        sethue("blue")
        Luxor.circle(pos[1, 1]*scale_fac, pos[1, 3]*scale_fac, 2.5, :stroke)
        sethue("black")
        Luxor.circle(pos[2, 1]*scale_fac, pos[2, 3]*scale_fac, 2.5, :stroke)
    end
end


function main(num_sids)
    # Output dir
    out_dir = joinpath(@__DIR__, "output")
    make_path(out_dir)


    # Loop through trials and overlay ground truth with the posterior simulation
    for trial = 0:209  # 209
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
        trace_df = trace_df[trace_df.t .== maximum(trace_df.t), :]  # get MAP at last time step
        sort!(trace_df, (:log_score), rev=(true))  # first entries have higher log scores
        trace_df = trace_df[1:num_sids, :]  # take num_sids best-scoring entries
        curr_state = fill(nothing, num_sids)
        curr_pos = fill(pos[1, :, :], num_sids)
        scene = Scene(scene_data, first(size(pos)), gm.run_mc_trace, 0.2)
        addr = :pos


        # Visualize ground truth
        for t = 1:first(size(pos))
            # Initialize drawing
            Drawing(750, 300, joinpath(trial_out_dir, @sprintf("time_%02d.png", t)))
            origin()
            translate(Point(0.,100.))
            background("white")

            # Draw the ground truth positions
            draw_groundtruth(scene_data, pos[t, :, :])

            # Draw the current state of the posterior simulation
            draw_simulation(curr_pos)

            # Simulate one step using the forward model and inferred latents
            params = [t, scene, prior, addr]
            (curr_state, curr_pos) = simulate_latents(generative_model, curr_state, params, trace_df)
            finish()
        end
        println("Done with trial $trial.")
    end
end

main(num_sids)
