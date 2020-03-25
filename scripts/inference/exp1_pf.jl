""""""
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
    parsed_args = parse_commandline()


    dataset_name = basename(args["dataset"])
    idx = args["idx"]
    out_dir = "traces/$dataset_name"
    out = "$out_dir/$(idx).jld2"
    isdir(dataset_name) || mkdir(out_dir)
    args["reset"] && isfile(out) && rm(out)

    run_exp1_trial(args["dataset"], idx, args["particles"],
                   args["obs_noise"], out)
    return nothing
end

main();
