using ArgParse
using GalileoRamp
using Base.Filesystem

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--chains", "-c"
        help = "The number of chains to run"
        arg_type = Int
        default = 1

        "--restart"
        help = "an option without argument, i.e. a flag"
        action = :store_true

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

function main()
    args = parse_commandline()
   
    dataset_name = first(splitext(basename(args["dataset"])))
    idx = args["idx"]
    particles = args["particles"]
    obs_noise = args["obs_noise"]
    out_dir = "/traces/$(dataset_name)_p_$(particles)_n_$(obs_noise)"
    out = "$out_dir/$(idx).jld2"
    isdir(out_dir) || mkdir(out_dir)
    isfile(out) && rm(out)

    seq_inference(args["dataset"], idx, args["particles"],
                  args["obs_noise"]; out = out, resume = true)
    return nothing
end

main();
