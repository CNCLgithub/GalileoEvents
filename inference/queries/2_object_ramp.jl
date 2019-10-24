"""
Describes the posteriors defined for Exp1
"""

using Gen
using PyCall
using Gen_Compose

forward_model = pyimport("galileo_ramp.world.simulation.forward_model");

struct Scene
    data::Dict
    n_frames::Int
end;

@gen function generative_model(prior::DeferredPrior, scene::Scene)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    t = scene.n_frames
    data["gravity"] = @trace(draw(prior, :gravity))
    data["objects"]["A"]["friction"] = @trace(draw(prior, :ramp_friction))

    state = forward_model.simulate(data, t, objs = scene.objects)
    pos = state[1]
    lin_vels = state[4]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, scene.obs_noise), :pos)
    return pos,lin_vels
end


latents = [:gravity, :ramp_friction]
prior = Gen_Compose.DeferredPrior(latents,
                                  [StaticDistribution{Float64}(uniform, (0.1, 20)),
                                   StaticDistribution{Float64}(uniform, (0.001, 0.999))])

function run_inference(scene_data, positions)
    scene = Scene(trial_data, length(positions))
    observations = Gen.choice_map()
    set_value!(observations, positions, :pos)
    query = StaticQuery(latents,
                        (scene,),
                        prior,
                        generative_model,
                        observations)
    procedure = MH()
    iterations = 100
    results = static_monte_carlo(procedure, query, iterations)
end

