export
    optimize_cuda_launch_config!

const CUDA_LAUNCH_AUTOTUNE_CACHE_RESET_HOOK = Ref{Any}(nothing)

#=
    cuda_launch_config(system)

Return the stored CUDA launch overrides for `system`.

The returned [`CUDALaunchConfig`](@ref) is consulted by Molly's tiled CUDA
pairwise kernels whenever they choose launch parameters for that specific
[`System`](@ref).
=#
cuda_launch_config(sys::System) = sys.launch_config

#=
    set_cuda_launch_config!(system, config::CUDALaunchConfig)
    set_cuda_launch_config!(system; kwargs...)

Set per-system CUDA launch overrides for Molly's tiled CUDA pairwise kernels.

This mutates `system.launch_config`. The new configuration is used by subsequent
CUDA force, tile-search, and energy launches for that system until
[`reset_cuda_launch_config!`](@ref) is called.
=#
function set_cuda_launch_config!(sys::System, config::CUDALaunchConfig)
    sys.launch_config = config
    return config
end

set_cuda_launch_config!(sys::System; kwargs...) =
    set_cuda_launch_config!(sys, CUDALaunchConfig(; kwargs...))

#=
    reset_cuda_launch_config!(system)

Clear any explicit per-system CUDA launch overrides and return to automatic
launch selection for the tiled CUDA pairwise kernels.
=#
reset_cuda_launch_config!(sys::System) = set_cuda_launch_config!(sys, CUDALaunchConfig())

"""
    optimize_cuda_launch_config!(system)

Autotune and cache CUDA launch overrides for the given system.

On CUDA systems, this benchmarks a small launch-candidate set for Molly's tiled
pairwise kernels, caches the result for the current device/system signature,
and writes the winning values into `system.launch_config`.
A fallback is provided for non-CUDA systems that does nothing.
"""
function optimize_cuda_launch_config!(sys)
    return nothing
end

@inline function maybe_optimize_cuda_launch_config!(sys; enabled::Bool=true)
    enabled || return nothing
    return optimize_cuda_launch_config!(sys)
end

#=
    reset_cuda_launch_autotune_cache!()

Clear any cached CUDA autotuning results.

This does not modify the currently active [`CUDALaunchConfig`](@ref); it only
clears the process-local cache used by [`optimize_cuda_launch_config!`](@ref).
On non-CUDA systems this is a no-op.
=#
function reset_cuda_launch_autotune_cache!()
    hook = CUDA_LAUNCH_AUTOTUNE_CACHE_RESET_HOOK[]
    hook === nothing && return nothing
    return hook()
end
