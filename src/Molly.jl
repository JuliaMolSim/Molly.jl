module Molly

import BioStructures # Imported to avoid clashing names
using CellListMap
using ChainRulesCore
import Chemfiles
using Colors
using Combinatorics
using CUDA
if has_cuda_gpu()
    CUDA.allowscalar(false)
end

using AMDGPU
if has_rocm_gpu()
    AMDGPU.allowscalar(false)
end

using DataStructures
using Distances
using Distributions
using EzXML
using FLoops
using ForwardDiff
using KernelDensity
using NearestNeighbors
using Reexport
using Requires
using Unitful
using UnitfulChainRules
using Zygote

@reexport using AtomsBase
@reexport using StaticArrays
@reexport using Unitful

using LinearAlgebra
using Random
using SparseArrays
using Statistics

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
include("interactions/harmonic_position_restraint.jl")
include("interactions/harmonic_bond.jl")
include("interactions/morse_bond.jl")
include("interactions/fene_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/cosine_angle.jl")
include("interactions/periodic_torsion.jl")
include("interactions/rb_torsion.jl")
include("interactions/implicit_solvent.jl")
include("energy.jl")
include("constraints.jl")
include("simulators.jl")
include("coupling.jl")
include("neighbors.jl")
include("loggers.jl")
include("analysis.jl")
include("setup.jl")
include("chain_rules.jl")
include("zygote.jl")
include("gradients.jl")

function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("makie.jl")
end

end
