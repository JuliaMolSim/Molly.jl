# CUDA launch configuration

export CUDALaunchConfig,
       cuda_launch_config,
       set_cuda_launch_config!,
       reset_cuda_launch_config!,
       reset_cuda_launch_autotune_cache!,
       optimize_cuda_launch_config!

"""
    CUDALaunchConfig(; force_block_y=nothing, force_maxregs=nothing,
                       tile_threads=nothing, energy_block_y=nothing)

Optional launch overrides for Molly's tiled CUDA pairwise kernels.

Passing `nothing` for a field leaves that choice on the automatic path.

# Fields
- `force_block_y`: number of tile rows processed per force-kernel block in the
  tiled pairwise force path.
- `force_maxregs`: optional `maxregs` cap for the tiled pairwise force kernel.
- `tile_threads`: `(threads_x, threads_y)` launch shape for the tile-finding
  kernel that scans the upper-triangular tile matrix.
- `energy_block_y`: number of tile rows processed per energy-kernel block in
  the tiled pairwise energy path.
"""
struct CUDALaunchConfig
    force_block_y::Union{Nothing, Int}
    force_maxregs::Union{Nothing, Int}
    tile_threads::Union{Nothing, NTuple{2, Int}}
    energy_block_y::Union{Nothing, Int}
end

function CUDALaunchConfig(;
                          force_block_y=nothing,
                          force_maxregs=nothing,
                          tile_threads=nothing,
                          energy_block_y=nothing)
    force_block_y === nothing || force_block_y > 0 ||
        throw(ArgumentError("force_block_y must be positive or nothing"))
    force_maxregs === nothing || force_maxregs > 0 ||
        throw(ArgumentError("force_maxregs must be positive or nothing"))
    energy_block_y === nothing || energy_block_y > 0 ||
        throw(ArgumentError("energy_block_y must be positive or nothing"))

    if tile_threads !== nothing
        length(tile_threads) == 2 || throw(ArgumentError("tile_threads must have length 2"))
        all(>(0), tile_threads) || throw(ArgumentError("tile_threads entries must be positive"))
    end

    return CUDALaunchConfig(force_block_y, force_maxregs, tile_threads, energy_block_y)
end

const CUDA_LAUNCH_CONFIG = Ref(CUDALaunchConfig())
const CUDA_LAUNCH_AUTOTUNE_CACHE_RESET_HOOK = Ref{Any}(nothing)

"""
    cuda_launch_config()

Return the currently active global CUDA launch overrides.

The returned [`CUDALaunchConfig`](@ref) is consulted by Molly's tiled CUDA
pairwise kernels whenever they choose launch parameters.
"""
cuda_launch_config() = CUDA_LAUNCH_CONFIG[]

"""
    set_cuda_launch_config!(config::CUDALaunchConfig)
    set_cuda_launch_config!(; kwargs...)

Set global CUDA launch overrides for Molly's tiled CUDA pairwise kernels.

This updates process-global state. The new configuration is used by subsequent
CUDA force, tile-search, and energy launches until
[`reset_cuda_launch_config!`](@ref) is called.
"""
function set_cuda_launch_config!(config::CUDALaunchConfig)
    CUDA_LAUNCH_CONFIG[] = config
    return config
end

set_cuda_launch_config!(; kwargs...) = set_cuda_launch_config!(CUDALaunchConfig(; kwargs...))

"""
    reset_cuda_launch_config!()

Clear any explicit CUDA launch overrides and return to automatic launch
selection for the tiled CUDA pairwise kernels.
"""
reset_cuda_launch_config!() = set_cuda_launch_config!(CUDALaunchConfig())

"""
    optimize_cuda_launch_config!(system)

Autotune and cache CUDA launch overrides for the given system.

On CUDA systems, this benchmarks a small launch-candidate set for Molly's tiled
pairwise kernels, caches the result for the current device/system signature,
and writes the winning values into the global launch override state.
A fallback is provided for non-GPU systems that does nothing.
"""
function optimize_cuda_launch_config!(sys)
    return nothing
end

"""
    reset_cuda_launch_autotune_cache!()

Clear any cached CUDA autotuning results.

This does not modify the currently active [`CUDALaunchConfig`](@ref); it only
clears the process-local cache used by [`optimize_cuda_launch_config!`](@ref).
On non-CUDA systems this is a no-op.
"""
function reset_cuda_launch_autotune_cache!()
    hook = CUDA_LAUNCH_AUTOTUNE_CACHE_RESET_HOOK[]
    hook === nothing && return nothing
    return hook()
end
