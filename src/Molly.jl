module Molly

using ProgressMeter
#using BioStructures
#import BioStructures.writepdb

using LinearAlgebra: norm, normalize, dot, ×
using StaticArrays

include("setup.jl")
include("md.jl")
#include("analysis.jl")

end
