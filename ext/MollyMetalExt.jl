"""
    MollyMetalExt

Metal extension for Molly.jl.

The GPU code of Molly, including the tiled pairwise kernels used with
[`GPUNeighborFinder`](@ref) and their autotuner, is written with KernelAbstractions and
KernelInterface and runs on any GPU backend that supports it.
This extension only holds what is specific to Metal:
- letting the GPU kernels use the atomics of Metal, which enables the tiled kernels of
  `GPUNeighborFinder` on Apple GPUs,
- compiling the tiled pairwise kernels with fast math (for `Float32`) and keeping the
  compiled kernels,
- passing the forces on the j atoms of a tile between the lanes of a SIMD-group with
  rotations and evaluating every iteration of the loop over the pairs of a tile, which
  are faster on Apple GPUs,
- the result of the displacement check of the tile list in shared memory that the device
  writes to, and an event that the host waits on.
"""
module MollyMetalExt

using Molly
using Metal
using Metal: @device_override, simd_shuffle_and_fill_up

## Atomics

#=
The kernels of Molly only use atomics on device memory: relaxed adds, loads and fetch-adds
of `Int32` and `Float32` values, and the atomic adds of the kernels written with
KernelAbstractions, which are sequentially consistent. Metal supports these for device
memory from Metal 3.0, with the ordered ones bracketed by fences before Metal 4.1, which
Metal.jl implements for every Metal version it targets (from Metal 3.2, as it requires
macOS 15). Metal.jl only reports
`KernelInterface.supports_atomics` from Metal 4.1 (macOS 27), which also adds atomics on
floats in threadgroup memory that Molly does not use. Without this, the tiled kernels of
`GPUNeighborFinder` are not used on Metal.
=#
Molly.supports_kernel_atomics(::MetalBackend) = true

## Compilation

#=
The tiled pairwise kernels are compiled with fast math for `Float32`, as on CUDA, which
changes the forces by about the rounding error of `Float32`. `MetalBackend` has no compiler
settings, so fast math is a compiler option.
=#
Molly.tile_kernel_config(backend::MetalBackend, ::Type{Float32}, maxregs) =
    (backend, (; fastmath=true))

#=
The compiled kernels with compiler options are kept like those without, see
`Molly.cached_kernel_function`, since the options are always given on Metal. Looking up the
kernel in the compilation cache at every launch otherwise takes longer on the host than
some of the kernels take to run.
=#
function Molly.tile_kernel(backend::MetalBackend, f::F, args::Tuple,
                           options::NamedTuple) where F
    return Molly.cached_kernel_function(backend, f, args, Val(options))
end

function Molly.tile_kernel(backend::MetalBackend, f::F, args::Tuple,
                           options::NamedTuple{()}) where F
    return Molly.cached_kernel_function(backend, f, args, Val(options))
end

## Tiled kernels

#=
The forces on the j atoms of a tile are passed between the lanes of a SIMD-group by
rotating them, see `Molly.tile_rotate`. `simd_shuffle_and_fill_up` with the values
themselves as the filling rotates them, and is much faster on Apple GPUs than
`simd_shuffle` from a lane that differs between the lanes.
=#
@device_override @inline function Molly.tile_rotate(val::SVector{D, Float32}, m, lane,
                                                    ::Val{32}) where D
    delta = m % Int16
    return SVector{D, Float32}(ntuple(k -> simd_shuffle_and_fill_up(@inbounds(val[k]),
                                                    @inbounds(val[k]), delta, Int16(32)),
                                      Val(D)))
end

#=
Every iteration of the loop over the pairs of a full tile is evaluated rather than only
those in which a lane of the SIMD-group has a pair within the cutoff, see
`Molly.tile_evaluates_pairs`. The iterations are those that have a pair within the cutoff
of the tile list, so few of them have no pair to evaluate.
=#
@device_override @inline Molly.tile_evaluates_pairs(condition::Bool, ::Val{32}) = true

## Displacement check

#=
The result of a check of the tile list is written by the device to shared memory, which the
host reads without a copy once an event signalled after the check has completed, as on
CUDA. Otherwise the copy of the result waits for all the work queued so far, which includes
the force kernel launched after the check, and the device then waits while the host queues
the work of the rest of the step. `event` is the event of the last check and the value it
is signalled with, or `nothing` once it has been waited for.
=#
mutable struct CheckEvent
    event::Union{Nothing, Metal.MTL.MTLSharedEvent}
    value::UInt64
end

function Molly.check_result_arrays(::MetalBackend, ::Type{T}, n::Integer) where T
    result = MtlArray{T, 1, Metal.SharedStorage}(undef, n)
    result_host = unsafe_wrap(Array, result)
    fill!(result_host, zero(T))
    return result, result_host, CheckEvent(nothing, UInt64(0))
end

# `record_event` submits the work queued so far, which ends with the check, and signals the
#   event once it has completed
function Molly.record_check!(backend::MetalBackend, ev::CheckEvent, scratch)
    ev.event, ev.value = Molly.KI.record_event(backend)
    return scratch
end

#=
The work queued since the check, such as the force kernel, is submitted first, so that the
device runs it while the host waits for the check and then queues the next work. Metal.jl
otherwise keeps it until more work is queued or the host synchronizes. The host spins on
the event, and after a second synchronizes, which reports errors of the device.
=#
function Molly.wait_check_result!(backend::MetalBackend, ev::CheckEvent, scratch)
    event = ev.event
    if !isnothing(event)
        Molly.KI.record_event(backend)
        if event.signaledValue < ev.value
            t_start = time_ns()
            while event.signaledValue < ev.value
                ccall(:jl_cpu_pause, Cvoid, ())
                # Lets other threads collect garbage while this one waits
                ccall(:jl_gc_safepoint, Cvoid, ())
                if time_ns() - t_start > 1_000_000_000
                    Metal.synchronize()
                    break
                end
            end
        end
        ev.event = nothing
    end
    return scratch.result_host
end

end
