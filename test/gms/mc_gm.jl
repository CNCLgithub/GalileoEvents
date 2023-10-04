using Gen
using GalileoEvents

mass_ratio = 2.0
obj_frictions = (0.3, 0.3)
obj_positions = (0.5, 1.5)

mprior = MaterialPrior([unknown_material])
pprior = PhysPrior((3.0, 10.0), # mass
                   (0.5, 10.0), # friction
                   (0.2, 1.0))  # restitution
obs_noise = 0.05
t = 120

function forward_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    mc_params = MCParams(client, [a,b], mprior, pprior, obs_noise)
    trace, _ = Gen.generate(mc_gm, (t, mc_params))
    #display(get_choices(trace))
end


function update_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    mc_params = MCParams(client, [a,b], mprior, pprior, obs_noise)
    trace, _ = Gen.generate(mc_gm, (t, mc_params))

    addr = :prior => :objects => 1 => :mass
    cm = Gen.choicemap(addr => trace[addr] + 3)
    trace2, _ = Gen.update(trace, cm)

    # compare final positions
    t=120
    pos1 = Vector(get_retval(trace)[t].bullet_state.kinematics[1].position)
    pos2 = Vector(get_retval(trace2)[t].bullet_state.kinematics[1].position)
    @assert pos1 != pos2

    return trace, trace2
end

function main()
    forward_test()
    update_test()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end