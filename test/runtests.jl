using Molly
using Aqua
import AtomsBase: Periodic, AbstractSystem
using AtomsBaseTesting
import BioStructures # Imported to avoid clashing names
using CUDA
using FiniteDifferences
using ForwardDiff
import SimpleCrystals
using Zygote

using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using Test

@warn "This file does not include all the tests for Molly.jl, " *
      "see the test directory for more"

# Allow testing of particular components
const GROUP = get(ENV, "GROUP", "All")
if GROUP in ("Protein", "Zygote", "NotZygote")
    @warn "Only running $GROUP tests as GROUP is set to $GROUP"
end

# Some CPU gradient tests give memory errors on CI
const running_CI = haskey(ENV, "CI")
if running_CI
    @warn "The visualization tests and some CPU gradient tests will not be run as this is CI"
end

# GLMakie doesn't work on CI or when running tests remotely
const run_visualize_tests = !running_CI && get(ENV, "VISTESTS", "1") != "0"
if run_visualize_tests
    using GLMakie
    @info "The visualization tests will be run as this is not CI"
elseif !running_CI && get(ENV, "VISTESTS", "1") == "0"
    @warn "The visualization tests will not be run as VISTESTS is set to 0"
end

const run_parallel_tests = Threads.nthreads() > 1
const n_threads_list = run_parallel_tests ? (1, Threads.nthreads()) : (1,)
if run_parallel_tests
    @info "The parallel tests will be run as Julia is running on $(Threads.nthreads()) threads"
else
    @warn "The parallel tests will not be run as Julia is running on 1 thread"
end

# Allow CUDA device to be specified
const DEVICE = get(ENV, "DEVICE", "0")

const run_gpu_tests = get(ENV, "GPUTESTS", "1") != "0" && CUDA.functional()
const gpu_list = run_gpu_tests ? (false, true) : (false,)
if run_gpu_tests
    device!(parse(Int, DEVICE))
    @info "The GPU tests will be run on device $DEVICE"
else
    @warn "The GPU tests will not be run as a CUDA-enabled device is not available"
end

const data_dir = normpath(@__DIR__, "..", "data")
const ff_dir     = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

const temp_fp_pdb = tempname(cleanup=true) * ".pdb"
const temp_fp_viz = tempname(cleanup=true) * ".mp4"

if GROUP in ("All", "NotZygote")
    # Some failures due to dependencies but there is an unbound args error
    Aqua.test_all(
        Molly;
        ambiguities=(recursive=false),
        unbound_args=false,
    )

    include("basic.jl")
    include("interactions.jl")
    include("minimization.jl")
    include("simulation.jl")
    include("agent.jl")
end

if GROUP in ("All", "Protein", "NotZygote")
    include("protein.jl")
end

if GROUP in ("All", "Zygote")
    include("zygote.jl")
end
