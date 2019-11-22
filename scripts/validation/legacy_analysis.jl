using CSV
using Glob
using JSON
using DataFrames

latents = [:friction_a, :friction_b, :friction_ground]


function extract_top_k(trace_path::AbstractString, k::Integer)
    df = sort(CSV.read(trace_path), :log_score, rev = true)
    first(df, k)
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
    trace_path = joinpath(proj_path, "traces", "match_legacy_mh_mass")
    traces = glob("trial_*.csv", trace_path)
    df = DataFrame()
    for trace in traces
        base_name = replace(basename(trace), "_trace.csv" => ".json")
        gt_json  = joinpath(proj_path, "scenes", "legacy_converted",
                        base_name)
        append!(df, process_trace(trace, gt_json))
    end

    CSV.write("$(trace_path)_summary_map.csv", df)
end

main()
