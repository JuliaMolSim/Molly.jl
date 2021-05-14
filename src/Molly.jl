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

@reexport using StaticArrays

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
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("makie.jl")
end

end
