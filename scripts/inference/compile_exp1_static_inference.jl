using CSV
using Glob
using JSON
using DataFrames


function extract_top_k(trace_path::AbstractString, k::Integer)
    df = sort(CSV.read(trace_path), :log_score, rev = true)
end

"""
Determines inference converges wrt to the GT.
Takes the top `k` estimates
"""
function process_trace(trace_path::AbstractString, scene_json::AbstractString,
                       k::Integer = 20)

    df = extract_top_k(trace_path, k)
    scene_data = Dict()
    open(scene_json, "r") do f
        scene_data=JSON.parse(f)["scene"]
    end
    # add gt values
    gt_mass_a = scene_data["objects"]["A"]["mass"]
    gt_mass_b = scene_data["objects"]["B"]["mass"]
    gt_mass_ratio = gt_mass_a / gt_mass_b

    df.trial_id = basename(trace_path)
    df.gt_mass_a = fill(gt_mass_a, size(df, 1))
    df.gt_mass_b = fill(gt_mass_b, size(df, 1))
    df.gt_mass_ratio = fill(gt_mass_ratio, size(df, 1))

    # add estimates
    df.mass_a = df.density_a * scene_data["objects"]["A"]["volume"]
    df.mass_b = df.density_b * scene_data["objects"]["B"]["volume"]
    df.mass_ratio = df.mass_a ./ df.mass_b
    return df
end


function main()
    proj_path = "../data/galileo-ramp"
    trace_path = joinpath(proj_path, "traces", "exp1_static_inference")
    traces = glob("trial_*.csv", trace_path)
    df = DataFrame()
    for idx = 0:209, tidx = 0:3
        trace = "$(trace_path)/trial_$(idx)_$(tidx)_trace.csv"
        gt_json  = joinpath(proj_path, "scenes", "legacy_converted",
                        "trial_$(idx).json")
        chunk = process_trace(trace, gt_json)
        chunk.trial_id = idx
        if idx < 120
            group = floor(idx/2)
            congruent = iseven(idx)
        else
            group = idx
            congruent = true
        end
        chunk[:group] = group
        chunk[:congruent] = congruent
        append!(df, chunk)
    end

    CSV.write("$(trace_path)_summary.csv", df)
    map_df = by(df, [:trial_id, :t],
                d ->  DataFrame(d[argmax(d[:, :log_score]),:]))
    CSV.write("$(trace_path)_summary_map.csv", map_df)
end

main()
