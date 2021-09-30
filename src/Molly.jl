module Molly

using BioStructures
using Colors
using CUDA
using Distances
using Distributions
using KernelDensity
using NearestNeighbors
using ProgressMeter
using Reexport
using Requires
using Unitful

@reexport using StaticArrays
@reexport using Unitful

using Base.Threads
using LinearAlgebra
using SparseArrays

include("types.jl")
include("cutoffs.jl")
include("setup.jl")
include("spatial.jl")
include("forces.jl")
include("simulators.jl")
include("thermostats.jl")
include("neighbors.jl")
include("loggers.jl")
include("analysis.jl")

function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("makie.jl")
end

end
