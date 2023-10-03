using FillArrays

export GMParams,
    GMState,
    Material,
    Iron,
    Brick,
    Wood,
    UnknownMaterial,
    iron,
    brick,
    wood,
    unknown_material,
    MaterialPrior,
    PhysPrior


################################################################################
# Common
################################################################################

"Parameters defining model behavior"
abstract type GMParams end

"Encapsulated state for a given model"
abstract type GMState end

abstract type Material end

struct Iron <: Material end
const iron = Iron()

struct Brick <: Material end
const brick = Brick()

struct Wood <: Material end
const wood = Wood()

struct UnknownMaterial <: Material end
const unknown_material = UnknownMaterial()

"""
A collection of materials and their associate prior weights

$(TYPEDEF)

---

$(TYPEDFIELDS)
"""
struct MaterialPrior
    materials::Vector{Material}
    material_weights::Vector{Float64}
end

"""
$(TYPEDSIGNATURES)

A uniform prior over given materials
"""
function MaterialPrior(ms::Vector{<: Material}) 
    n = length(ms)
    ws = Fill(1.0 / n, n)
    MaterialPrior(ms, ws)
end

"""
Parameterizes an object's prior distribtuion over physical properties.

$(TYPEDEF)

---

$(TYPEDFIELDS)
"""
struct PhysPrior
    mass::NTuple{2, Float64}
    friction::NTuple{2, Float64}
    restitution::NTuple{2, Float64}
end

include("mc_gm.jl")
# include("cp_gm.jl")
