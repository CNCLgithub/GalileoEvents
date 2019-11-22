using ArgParse
using PyCall

include("../../inference/visualize/visualize.jl")

np = pyimport("numpy")
gm = pyimport("galileo_ramp.world.simulation.exp2_physics");

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--directory"
            default = "legacy_converted"
        "scene"
            help = "Path to the scene of interest"
            required = true
    end

    return parse_args(s)
end

function main()
    # args = parse_commandline()
    args = Dict(("scene" => "../data/galileo-ramp/scenes/legacy_converted/trial_88.json"))
    scene_data = Dict()
    open(args["scene"], "r") do f
        scene_data=JSON.parse(f)["scene"]
    end
    gt_file = replace(args["scene"], ".json" => "_pos.npy")
    pos = np.load(gt_file)
    pos = permutedims(pos, [2, 1, 3])
    n_frames = first(size(pos))
    fps = 60
    dur = n_frames * 1/fps
    sim = Array{Float64, 4}(undef, 1, size(pos)...)
    sim[1,: ,:, :] =  gm.run_full_trace(scene_data, ["A", "B"], dur,
                                        fps = fps,
                                        time_scale = 10.0)[1]
    # sim[2,: ,:, :] =  gm.run_full_trace(scene_data, ["A", "B"], dur,
    #                                     fps = fps,
    #                                     time_scale = 10.0)[1]
    # sim[1,: ,:, :] =  gm.run_full_trace(scene_data, ["A", "B"], n_frames,
    #                                     fps = 24)[1]
    outpath = "test"
    visualize(scene_data, pos, sim, outpath)
end

main()
