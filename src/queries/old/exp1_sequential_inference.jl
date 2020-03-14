"""
Attempts to infer physical parameters in the Exp2 physics engine
that match observations generated from Exp1 physics engine
"""

using Gen
using CSV
using JSON
using PyCall
using Gen_Compose
using ProfileView

include("../dist.jl")

np = pyimport("numpy")
gm = pyimport("physics.world.simulation.exp2_physics");


Gen.load_generated_functions()

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
    scene = Scene(scene_data, t, gm.run_mc_trace,
                  density_a, density_b, 0.2)

    args = [(i, scene) for i in 1:t]
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
    n_particles = 2
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    @time results = sequential_monte_carlo(procedure, query)
    # @profview sequential_monte_carlo(procedure, query)
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
    t = 5
    run_inference(scene_data, pos[1:t, :, :], "test")
end

