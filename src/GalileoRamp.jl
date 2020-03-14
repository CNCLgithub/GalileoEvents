module GalileoRamp

using Gen
using PyCall

const physics = PyNULL()
function __init__()
    copy!(physics, pyimport("physics.world"))
end


include("distributions.jl")
include("gms/gms.jl")
include("analysis.jl")

end # module
