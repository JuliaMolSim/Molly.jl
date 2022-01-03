module Molly

using BioStructures
using CellListMap
using ChainRulesCore
import Chemfiles # Imported to avoid clashing names
using Colors
using Combinatorics
using CUDA
using Distances
using Distributions
using EzXML
using ForwardDiff
using KernelDensity
using NearestNeighbors
import NNlib
using ProgressMeter
using Reexport
using Requires
using Unitful
using Zygote

@reexport using AtomsBase
@reexport using StaticArrays
@reexport using Unitful

using Base.Threads
using LinearAlgebra
using SparseArrays

include("types.jl")
include("cutoffs.jl")
include("spatial.jl")
include("force.jl")
include("interactions/lennard_jones.jl")
include("interactions/soft_sphere.jl")
include("interactions/mie.jl")
include("interactions/coulomb.jl")
include("interactions/coulomb_reaction_field.jl")
include("interactions/gravity.jl")
include("interactions/harmonic_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/periodic_torsion.jl")
include("interactions/rb_torsion.jl")
include("energy.jl")
include("simulators.jl")
include("coupling.jl")
include("neighbors.jl")
include("loggers.jl")
include("analysis.jl")
include("chain_rules.jl")
include("zygote.jl")
include("setup.jl")

function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("makie.jl")
end

end
