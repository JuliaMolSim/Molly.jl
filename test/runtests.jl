using Molly
using ParallelTestRunner
using Suppressor

# Suppress warnings for unavailable backends, warn later
@suppress_err using AMDGPU
@suppress_err using CUDA
@suppress_err using Metal
@suppress_err using oneAPI

const n_threads_per_job = 4

const run_visualize_tests = get(ENV, "VISTESTS", "1") != "0"
if !run_visualize_tests
    @warn "The visualization tests will not be run as VISTESTS is set to 0"
end

const run_gpu_tests = get(ENV, "GPUTESTS", "1") != "0"

const DEVICE = parse(Int, get(ENV, "DEVICE", "0")) # Allow GPU device to be specified

if run_gpu_tests && CUDA.functional()
    @info "The CUDA tests will be run on device $DEVICE"
else
    @warn "The CUDA tests will not be run as a CUDA-enabled device is not available"
end

if run_gpu_tests && AMDGPU.functional()
    @info "The AMDGPU tests will be run on device $amd_device"
else
    @warn "The AMDGPU tests will not be run as a AMDGPU-enabled device is not available"
end

if run_gpu_tests && oneAPI.functional()
    @info "The oneAPI tests will be run on device $DEVICE"
else
    @warn "The oneAPI tests will not be run as a oneAPI-enabled device is not available"
end

if run_gpu_tests && Metal.functional()
    @info "The Metal tests will be run"
else
    @warn "The Metal tests will not be run as a Metal-enabled device is not available"
end

const init_code_block = quote
    using Molly
    using Molly: from_device, to_device, NaNSimulationError, ForceFieldXMLError,
                 MissingResidueTemplateError
    using Aqua
    import AtomsBase
    using AtomsBaseTesting
    import AtomsCalculators
    using BenchmarkTools
    import BioStructures
    import Chemfiles
    using Enzyme
    using FiniteDifferences
    using GPUArrays
    using HDF5
    using JET
    using JSON3
    using KernelAbstractions
    using KernelDensity
    using Lux
    using Measurements
    import SimpleCrystals
    using Suppressor

    @suppress_err using AMDGPU
    @suppress_err using CUDA
    @suppress_err using Metal
    @suppress_err using oneAPI

    using DelimitedFiles
    using LinearAlgebra
    using Random
    using Statistics
    using Test

    Enzyme.Compiler.VERBOSE_ERRORS[] = true

    const run_visualize_tests = $run_visualize_tests
    if run_visualize_tests
        import GLMakie
    end

    const n_threads_list = (1, $n_threads_per_job)

    const run_gpu_tests = $run_gpu_tests
    const run_cuda_tests  = run_gpu_tests && CUDA.functional()
    const run_metal_tests = run_gpu_tests && Metal.functional()
    const DEVICE = $DEVICE
    array_list = (Array,)

    if run_cuda_tests
        array_list = (array_list..., CuArray)
        CUDA.device!(DEVICE)
    end
    if run_gpu_tests && AMDGPU.functional()
        array_list = (array_list..., ROCArray)
        amd_device = (iszero(DEVICE) ? 1 : DEVICE)
        AMDGPU.device!(AMDGPU.device(amd_device))
    end
    if run_gpu_tests && oneAPI.functional()
        array_list = (array_list..., oneArray)
        oneAPI.device!(DEVICE)
    end
    if run_metal_tests
        # Metal only supports 32-bit precision, so MtlArray can not be added to array_list
        array_list_metal = (array_list..., MtlArray)
    else
        array_list_metal = array_list
    end

    const data_dir = normpath(@__DIR__, "..", "data")
    const ff_dir     = joinpath(data_dir, "force_fields")
    const openmm_dir = joinpath(data_dir, "openmm_6mrr")
end

# Allow @suppress_err to work in the quote block
const init_code = Expr(:toplevel, init_code_block.args...)

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    for k in keys(testsuite)
        if startswith(k, "extra") || startswith(k, "reference")
            delete!(testsuite, k)
        end
    end
end

runtests(
    Molly,
    ARGS;
    testsuite=testsuite,
    init_code=init_code,
    exeflags=["--threads=$n_threads_per_job"],
)
