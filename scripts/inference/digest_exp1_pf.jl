using GalileoRamp
using FileIO
using CSV
using Glob
using DataFrames
using Statistics

function process_trial(dataset_path::String,
                       trace_path::String,
                       trial::Int)

    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, tps) = get(dataset, trial)

    chain_paths = glob("$(trial)_c_*.jld2", "$(trace_path)")
    println(chain_paths)
    chain_paths = isempty(chain_paths) ? ["$(trace_path)/$(trial).jld2"] : chain_paths
    extracts = map(extract_chain, chain_paths)
    densities = hcat(map(e -> e["unweighted"][:ramp_density], extracts)...)
    merged = Dict("log_scores" => merge(hcat, extracts...)["log_scores"],
                  "unweighted" => Dict(:ramp_density => densities))
    df = to_frame(merged["log_scores"], merged["unweighted"])
    df = df[in.(df.t, Ref(tps)),:]
    df = aggregate(groupby(df, :t), mean)
    sort(df, :t)
end

particles = [100];
noises = [0.01, 0.0362, 0.05];
let model = 1
dfs = []
for ps in particles
    for noise in noises
        println(ps, noise)
        traces = "/traces/exp1_p_$(ps)_n_$(noise)"
        for i = 0:209
            t = process_trial("/databases/exp1.hdf5", traces, i)
            if i < 120
                t.scene = floor(i/2)
                t.congruent = (i % 2) == 0
            else
                t.scene = i - 60
                t.congruent = true
            end
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
