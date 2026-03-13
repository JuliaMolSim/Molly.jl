# CUDA launch configuration

export CUDALaunchConfig,
       cuda_launch_config,
       set_cuda_launch_config!,
       reset_cuda_launch_config!

"""
    CUDALaunchConfig(; force_block_y=nothing, force_maxregs=nothing,
                       tile_threads=nothing, energy_block_y=nothing)

Optional launch overrides for Molly's CUDA kernels.

Passing `nothing` for a field leaves that choice on the automatic path.
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

"""
    cuda_launch_config()

Return the currently active CUDA launch overrides.
"""
cuda_launch_config() = CUDA_LAUNCH_CONFIG[]

"""
    set_cuda_launch_config!(config::CUDALaunchConfig)
    set_cuda_launch_config!(; kwargs...)

Set global CUDA launch overrides for Molly's CUDA kernels.
"""
function set_cuda_launch_config!(config::CUDALaunchConfig)
    CUDA_LAUNCH_CONFIG[] = config
    return config
end

set_cuda_launch_config!(; kwargs...) = set_cuda_launch_config!(CUDALaunchConfig(; kwargs...))

"""
    reset_cuda_launch_config!()

Clear any explicit CUDA launch overrides and return to the automatic path.
"""
reset_cuda_launch_config!() = set_cuda_launch_config!(CUDALaunchConfig())
