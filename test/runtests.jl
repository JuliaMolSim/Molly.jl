using Molly
using Molly: from_device, to_device
using AMDGPU
using Aqua
import AtomsBase
using AtomsBaseTesting
import AtomsCalculators
using AtomsCalculators.AtomsCalculatorsTesting
using BenchmarkTools
import BioStructures
import Chemfiles
using CUDA
using Enzyme
using FiniteDifferences
using GPUArrays
using KernelDensity
using Measurements
using Metal
using oneAPI
import SimpleCrystals
using Suppressor

using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using Test

@warn "This file does not include all the tests for Molly.jl, " *
      "see the test directory for more"

# Allow testing of particular components
const GROUP = get(ENV, "GROUP", "All")
if GROUP in ("Protein", "Gradients", "NotGradients")
    @warn "Only running $GROUP tests as GROUP is set to $GROUP"
elseif GROUP != "All"
    error("Unrecognised test group, GROUP=$GROUP")
end

# Some CPU gradient tests give memory errors on CI
const running_CI = haskey(ENV, "CI")
if running_CI
    @warn "Some CPU gradient tests will not be run as this is CI"
end

const run_visualize_tests = get(ENV, "VISTESTS", "1") != "0"
if run_visualize_tests
    import GLMakie
else
    @warn "The visualization tests will not be run as VISTESTS is set to 0"
end

const run_parallel_tests = (Threads.nthreads() > 1)
const n_threads_list = (run_parallel_tests ? (1, Threads.nthreads()) : (1,))
if run_parallel_tests
    @info "The parallel tests will be run as Julia is running on $(Threads.nthreads()) threads"
else
    @warn "The parallel tests will not be run as Julia is running on 1 thread"
end

const run_gpu_tests = get(ENV, "GPUTESTS", "1") != "0"
# Allow GPU device to be specified
const DEVICE = parse(Int, get(ENV, "DEVICE", "0"))

const run_cuda_tests   = run_gpu_tests && CUDA.functional()
const run_rocm_tests   = run_gpu_tests && AMDGPU.functional()
const run_oneapi_tests = run_gpu_tests && oneAPI.functional()
const run_metal_tests  = run_gpu_tests && Metal.functional()

array_list = (Array,)

if run_cuda_tests
    array_list = (array_list..., CuArray)
    CUDA.device!(DEVICE)
    @info "The CUDA tests will be run on device $DEVICE"
else
    @warn "The CUDA tests will not be run as a CUDA-enabled device is not available"
end

if run_rocm_tests
    array_list = (array_list..., ROCArray)
    amd_device = (iszero(DEVICE) ? 1 : DEVICE)
    AMDGPU.device!(AMDGPU.device(amd_device))
    @info "The AMDGPU tests will be run on device $amd_device"
else
    @warn "The AMDGPU tests will not be run as a AMDGPU-enabled device is not available"
end

if run_oneapi_tests
    array_list = (array_list..., oneArray)
    oneAPI.device!(DEVICE)
    @info "The oneAPI tests will be run on device $DEVICE"
else
    @warn "The oneAPI tests will not be run as a oneAPI-enabled device is not available"
end

if run_metal_tests
    @info "The Metal tests will be run"
else
    @warn "The Metal tests will not be run as a Metal-enabled device is not available"
end

const data_dir = normpath(@__DIR__, "..", "data")
const ff_dir     = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

const temp_fp_dcd  = tempname(cleanup=true) * ".dcd"
const temp_fp_trr  = tempname(cleanup=true) * ".trr"
const temp_fp_pdb  = tempname(cleanup=true) * ".pdb"
const temp_fp_xyz  = tempname(cleanup=true) * ".xyz"
const temp_fp_mol2 = tempname(cleanup=true) * ".mol2"
const temp_fp_mp4  = tempname(cleanup=true) * ".mp4"

Enzyme.Compiler.VERBOSE_ERRORS[] = true

if GROUP in ("All", "NotGradients")
    # Some failures due to dependencies but there is an unbound args error
    Aqua.test_all(
        Molly;
        ambiguities=(recursive=false),
        unbound_args=false,
        piracies=false,
    )

    include("basic.jl")
    include("interactions.jl")
    include("minimization.jl")
    include("agent.jl")
    include("simulation.jl")
    include("bias.jl")
    include("coupling.jl")
    include("constraints.jl")
    include("analysis.jl")
end

if GROUP in ("All", "Protein", "NotGradients")
    include("protein.jl")
end

if GROUP in ("All", "Gradients")
    include("gradients.jl")
end
