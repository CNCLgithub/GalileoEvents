module GalileoEvents

using Accessors
using Gen
using Gen_Compose
using PhySMC
using PhyBullet
using PyCall
using Parameters
using DocStringExtensions

include("utils/utils.jl")
include("gms/gms.jl")
# include("procedures/procedures.jl")
# include("queries/queries.jl")
# include("analysis.jl")
# include("visualize/visualize.jl")

#################################################################################
# Load Gen functions
#################################################################################

@load_generated_functions

end # module
