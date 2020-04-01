using GalileoRamp
using FileIO
using CSV
using DataFrames



function process_trial(dataset_path::String,
                       trace_path::String,
                       trial::Int)

    dataset = GalileoRamp.galileo_ramp.Exp1Dataset(dataset_path)
    (scene, state, cols) = get(dataset, trial)

    chain_path = "$trace_path/$(trial).jld2"
    extracted = extract_chain(chain_path)

    df = to_frame(extracted["log_scores"], extracted["unweighted"],
                  exclude = [:ramp_pos])
    df = df[in.(df.t, Ref(cols)),:]
    if trial < 120
        df.scene = floor(trial/2)
        df.congruent = (trial % 2) == 0
    else
        df.scene = trial - 60
        df.congruent = true
    end
    sort(df, :t)
end

# process_trial("/databases/exp1.hdf5", "/traces", 1);

particles = 300;
noises = [0.05, 0.1, 0.8];
for noise in noises
    traces = "/traces/exp1_p_$(particles)_n_$(noise)"
    df = DataFrame()
    for i = 0:209
        t = process_trial("/databases/exp1.hdf5", traces, i)
        df = vcat(df, t)
    end
    df.particles = particles;
    df.noise = noise
    CSV.write("$(traces)/digest.csv", df)
end
