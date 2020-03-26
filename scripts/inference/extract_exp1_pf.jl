"""Extracts inference traces from chains for analysis"""

using GalileoRamp
using JLD2
using FileIO
using DataFrames

density_map = Dict(
    "density" => cm -> cm[:object_physics => 1 => :density]
)

function parse_chain(chain_path, time_points)
    chain = load(chain_path)
    e = extract_chain(chain, density_map)
    df = to_frame(e["log_scores"], e["unweighted"])
    filter!(row -> row[:t] in time_points, df)
    return df
end

function main():
    dataset_path = "/databases/exp1.hdf5"
    dataset = ...
    traces = "/traces/exp1_p_10"
    df = DataFrame()
    for i = 1:length(dataset)
        scene,_,tps = get(dataset, i)
        chain_path = "$(traces)/$i.jld2"
        chain_df = parse_chain(chain_path, tps)
        chain_df[:scene] = i
        df = join(df, chain_df, kind = :outer)
    end
    CSV.write(df, "$traces/summary.csv")
end


main();
