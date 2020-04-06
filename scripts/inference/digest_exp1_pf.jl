using GalileoRamp
using FileIO
using CSV
using DataFrames

function process_trial(dataset_path::String,
                       trace_path::String,
                       trial::Int)

    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, tps) = get(dataset, trial)

    chain_paths = glob("$trace_path/$(trial)_c*.jld2")
    extracts = map(extract_chain, chain_paths)
    merged = merge(hcat, extracts...)

    df = to_frame(extracted["log_scores"], extracted["unweighted"])
    df = df[in.(df.t, Ref(tps)),:]
    sort!(df, :t)
    aggregate(groupby(df, :t), mean)

    sort(df, :t)
end

particles = [4, 10, 300];
noises = [0.1, 0.2, 0.5];
model = 1;
df = DataFrame()
for ps in particles
    for noise in noises
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
            df = vcat(df, t)
        end
        model += 1
    end
    CSV.write("traces/exp1_digest.csv", df)
end
