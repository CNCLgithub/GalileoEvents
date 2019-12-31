"""
Attempts to infer physical parameters in the Exp2 physics engine
that match observations generated from Exp1 physics engine
"""

using Gen
using CSV
using JSON
using PyCall
using Gen_Compose
# import Cairo

include("../dist.jl")
# include("../procedures/static.jl")
# include("../visualize/plot_inference.jl")

np = pyimport("numpy")
gm = pyimport("galileo_ramp.world.simulation.exp2_physics");

struct Scene
    data::Dict
    n_frames::Int
    simulator::T where T<:Function
    density_a::Float64
    density_b::Float64
    obs_noise::Float64
end;

@gen function generative_model(state, t::Int, scene::Scene)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    data["objects"]["A"]["density"] = @trace(trunc_norm(scene.density_a,
                                                        3.0, 0., 12.), :density_a)
    data["objects"]["A"]["mass"] = data["objects"]["A"]["density"] * data["objects"]["A"]["volume"]
    data["objects"]["B"]["density"] = @trace(trunc_norm(scene.density_b,
                                                        3.0, 0., 12.), :density_b)
    data["objects"]["B"]["mass"] = data["objects"]["B"]["density"] * data["objects"]["B"]["volume"]
    new_state = scene.simulator(data, ["A", "B"], scene.n_frames * 1.0/60.0,
                                fps = 60, time_scale = 10.0)
    pos = new_state[1][end, :]
    @trace(Gen.broadcasted_normal(pos, fill(scene.obs_noise, scene.n_frames)),
           (:pos, t))
    return nothing
end

function run_inference(scene_data, positions, out_path,
                       iter::Int = 100)
    t = first(size(positions))

    # turn the observation into a vector of matrices (for SequentialQuery)
    obs = Vector{Matrix{Float64}}(undef, size(positions, 1))
    for i = 1:t
        obs[i] = positions[i, :, :]
    end
    observations = Gen.choicemap()
    set_value!(observations, :pos, obs)

    latents = [:density_a, :density_b]
    density_map = Dict()
    density_map["Wood"] = 1.0
    density_map["Brick"] = 2.0
    density_map["Iron"] = 8.0

    density_a = density_map[scene_data["objects"]["A"]["appearance"]]
    density_b = density_map[scene_data["objects"]["B"]["appearance"]]
    scene = Scene(scene_data, t, gm.run_full_trace,
                  density_a, density_b, 0.2)

    args = [(scene,) for _ in 1:t]
    query = SequentialQuery(latents,
                            generative_model,
                            args,
                            observations)
    mass_move = DynamicDistribution{Float64}(log_uniform, x -> (x, 0.1))
    moves = [mass_move
             mass_move]

    # the rejuvination will follow Gibbs sampling
    rejuv = gibbs_steps(moves, latents)

    # -----------------------------------------------------------
    # Define the inference procedure
    # In this case we will be using a particle filter
    #
    # Additionally, this will be under the Sequential Monte-Carlo
    # paradigm.
    n_particles = 10
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    @time results = sequential_monte_carlo(procedure, query)

    # plot = viz(results)
    # plot |> PNG("$(out_path)_trace.png",30cm, 30cm)
    df = to_frame(results)
    df.t = t
    CSV.write("$(out_path)_trace.csv", df)
    return df
end

function main()
    scene_json = "../../data/galileo-ramp/scenes/legacy_converted/trial_89.json"
    scene_pos = "../../data/galileo-ramp/scenes/legacy_converted/trial_89_pos.npy"
    scene_data = Dict()
    open(scene_json, "r") do f
        # dicttxt = readstring(f)
        scene_data=JSON.parse(f)["scene"]
    end
    println(scene_data["objects"]["A"]["mass"])
    println(scene_data["objects"]["B"]["mass"])
    pos = np.load(scene_pos)
    pos = permutedims(pos, [2, 1, 3])
    t = 50
    run_inference(scene_data, pos[1:t, :, :], "test")
end

