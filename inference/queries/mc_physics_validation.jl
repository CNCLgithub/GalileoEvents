"""
Runs inference over the `test_physics` scene
"""

using Gen
using PyCall
using Gen_Compose

# forward_model = pyimport("galileo_ramp.world.simulation.test_physics.py");

struct Scene
    data::Dict
    n_frames::Int
    simulator::T where T<:Function
end;

@gen function generative_model(prior::DeferredPrior, scene::Scene, t::Int)
    # Add the features from the latents to the scene descriptions
    data = {}
    data["friction"] = @trace(draw(prior, :friction))
    data["mass"] = @trace(draw(prior, :mass))

    state = scene.simulator(data, t)
    pos = state[1]
    lin_vels = state[4]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, scene.obs_noise), :pos)
    return pos,lin_vels
end


latents = [:mass, :friction]
prior = Gen_Compose.DeferredPrior(latents,
                                  [StaticDistribution{Float64}(uniform, (0.1, 10)),
                                   StaticDistribution{Float64}(uniform, (0.001, 0.999))])

function run_inference(scene_data, simulator, iterations = 100)
    scene = Scene(trial_data, length(positions), simulator)
    observations = Gen.choice_map()
    set_value!(observations, positions, :pos)
    query = SequentialQuery(latents,
                            (scene,),
                            prior,
                            generative_model,
                            observations)
    procedure = MH()
    results = static_monte_carlo(procedure, query, iterations)
end

