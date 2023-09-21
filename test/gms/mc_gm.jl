using Gen
using GalileoEvents

mass_ratio = 2.0
obj_frictions = (0.3, 0.3)
obj_positions = () # TODO fill in...

mprior = MaterialPrior([unknown_material])
pprior = PhysPrior((3.0, 10.0), # mass
                   (0.5, 10.0), # friction
                   (0.2, 1.0))  # restitution

t = 60

# TODO: this should evaluate without errors
function forward_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    mc_params = MCParams(client, [a,b], mprior, pprior)
    trace, _ = Gen.generate(t, mc_gm)
    display(get_choices(trace))
end


# TODO: use `Gen.update` to change a traces physical latents and
# compare the final positions (they should be different)
function update_test()
    client, a, b = ramp(mass_ratio, obj_frictions, obj_positions)
    mc_params = MCParams(client, [a,b], mprior, pprior)
    trace, _ = Gen.generate(t, mc_gm)

    addr = :prior => :objects => 1 => :mass
    cm = Gen.choicemap(addr => trace[addr] + 3.0)
    trace2 = Gen.update(trace, cm)


    # compare final positions
end
