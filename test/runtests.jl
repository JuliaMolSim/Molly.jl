using Molly
using Aqua
import BioStructures # Imported to avoid clashing names
using CUDA
using FiniteDifferences
using ForwardDiff
using Zygote

using Base.Threads
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using Test

@warn "This file does not include all the tests for Molly.jl due to CI time limits, " *
        "see the test directory for more"

# Allow testing of particular components
const GROUP = get(ENV, "GROUP", "All")
if GROUP == "Protein" || GROUP == "Zygote"
    @warn "Only running $GROUP tests as GROUP is set to $GROUP"
end

# Allow CUDA device to be specified
const DEVICE = get(ENV, "DEVICE", "0")

# GLMakie doesn't work on CI
run_visualize_tests = !haskey(ENV, "CI")
if run_visualize_tests
    using GLMakie
    @info "The visualization tests will be run as this is not CI"
else
    @warn "The visualization tests will not be run as this is CI"
end

run_parallel_tests = Threads.nthreads() > 1
if run_parallel_tests
    @info "The parallel tests will be run as Julia is running on $(Threads.nthreads()) threads"
else
    @warn "The parallel tests will not be run as Julia is running on 1 thread"
end

run_gpu_tests = CUDA.functional()
if run_gpu_tests
    device!(parse(Int, DEVICE))
    @info "The GPU tests will be run on device $DEVICE"
else
    @warn "The GPU tests will not be run as a CUDA-enabled device is not available"
end

CUDA.allowscalar(false) # Check that we never do scalar indexing on the GPU

data_dir = normpath(@__DIR__, "..", "data")
ff_dir = joinpath(data_dir, "force_fields")

temp_fp_pdb = tempname(cleanup=true) * ".pdb"
temp_fp_viz = tempname(cleanup=true) * ".mp4"

if GROUP == "All"
    # Some failures due to dependencies but there is an unbound args error
    Aqua.test_all(
        Molly;
        ambiguities=(recursive=false),
        unbound_args=false,
        undefined_exports=false,
    )

    include("basic.jl")
    include("minimization.jl")
    include("simulation.jl")
    include("agent.jl")
end

if GROUP == "All" || GROUP == "Protein"
    include("protein.jl")
end

if GROUP == "All" || GROUP == "Zygote"
    include("zygote.jl")
end
