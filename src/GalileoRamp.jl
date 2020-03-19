module GalileoRamp

using Gen
using Gen_Compose
using PyCall

const physics = PyNULL()
const galileo_ramp = PyNULL()
function __init__()
    copy!(physics, pyimport("physics.world"))
    copy!(galileo_ramp, pyimport("galileo_ramp.exp1_dataset"))
end

include("distributions.jl")
include("gms/gms.jl")
include("queries/queries.jl")
include("analysis.jl")

end # module
