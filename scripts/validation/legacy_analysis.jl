using CSV
using Glob
using DataFrames

latents = [:friction_a, :friction_b, :friction_ground]

"""
Determines inference converges wrt to the GT.
Takes the top `k` estimates
"""
function process_trace(trace_path, scene_json,
                       k = 20)
    df = CSV.read(trace_path)
    sort!(df, :log_score, rev = true)

    scene_data = Dict()
    open(scene_json, "r") do f
        scene_data=JSON.parse(f)["scene"]
    end
    gt_friction_a = scene_data["objects"]["A"]["friction"]
    gt_friction_b = scene_data["objects"]["B"]["friction"]
    gt_friction_ground = 0.5

    plots = map((l,gt) -> plot_latent(l, gt, df),
                latents,
                [gt_friction_a, gt_friction_b, gt_friction_ground])



end


function main()
    files = glob("*.csv", trace_path)
end

