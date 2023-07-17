


abstract type PhysicsGM end

abstract type SimState end


"""
     step(gm::PhysicsGM, st::SimState)::SimState

Performs a stateless evolution of the simulation state.
"""
function step(gm::PhysicsGM, st::StimState)
    sync!(gm, st)
    new_st = forward_step(gm)
end


"""
    sync!(gm::PhysicsGM, st::SimState)::Nothing

Synchronizes the context within `gm` using `st`.
"""
function sync! end



"""
    forward_step(gm::PhysicsGM)::SimState

Resolves physical interactions and obtains the next state representation.
"""
function forward_step end



# REVIEW: is useful or possible?
# """
#     is_synced(gm::PhysicsGM, st::SimState)::Bool

# Determines whether the gm
# """
# function is_synced end

include("bullet_gm.jl")
# include("mc_gm.jl")
include("cp_gm.jl")
