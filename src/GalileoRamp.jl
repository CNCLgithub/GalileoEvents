module GalileoRamp

using Gen
using Gen_Compose
using PyCall

const physics = PyNULL()
const galileo_ramp = PyNULL()
function __init__()
    # setup python imports
    copy!(physics, pyimport("ramp_physics"))
    copy!(galileo_ramp, pyimport("galileo_ramp.exp1_dataset"))

end

include("distributions.jl")
include("gms/gms.jl")
include("procedures/procedures.jl")
include("queries/queries.jl")
include("analysis.jl")
include("visualize/visualize.jl")

#################################################################################
# Load Gen functions
#################################################################################

@load_generated_functions

end # module
