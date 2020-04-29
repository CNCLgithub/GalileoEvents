using GalileoRamp
using FileIO
using CSV
using Glob
using DataFrames
using Statistics

function process_trial(dataset,
                       trace_path::String,
                       trial::Int)
    (scene, state, tps) = get(dataset, trial)
    csv = "$(trace_path)/$trial.csv"
    df = DataFrame(CSV.File(csv))
    sort(df, :t)
end

dataset_path = "/databases/exp1.hdf5"
dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
particles = [100];
noises = [0.01, 0.0362, 0.05];
let model = 1
dfs = []
for ps in particles
    for noise in noises
        println(ps, noise)
        traces = "/traces/exp1_p_$(ps)_n_$(noise)"
        for i = 0:119
            t = process_trial(dataset, traces, i)
            t.scene = floor(i/2)
            t.congruent = (i % 2) == 0
            t.particles = ps
            t.noise = noise
            t.model = model
            push!(dfs, t)
        end
        model += 1
    end
    df = vcat(dfs...)
    CSV.write("/traces/exp1_digest.csv", df)
end
end
