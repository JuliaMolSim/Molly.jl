"""
    MollyCUDAExt

CUDA extension for Molly.jl.

The GPU code of Molly, including the tiled pairwise kernels used with
[`GPUNeighborFinder`](@ref) and their autotuner, is written with KernelAbstractions and
KernelInterface and runs on any GPU backend that supports it.
This extension only holds what is specific to CUDA:
- compiling the pairwise kernels with fast math (for `Float32`), always inlining and
  an optional register limit,
- the CUDA context that compiled kernels belong to and the static shared memory limit,
- running the specific and general interactions on a second stream at the same time as
  the pairwise interactions,
- the result of the displacement check of the neighbor lists in pinned memory that the
  device writes to, and an event that the host waits on,
- device selection for replica exchange.
"""
module MollyCUDAExt

using Molly
using CUDA

function Molly.get_gpu_devices(::Val{true})
    return collect(CUDA.devices())
end

function Molly.set_gpu_device!(gpu_id, ::Val{true})
    CUDA.device!(gpu_id)
end

## Compilation

# Whether to compile the pairwise kernels with fast math
# Restricted to Float32, where the extra error is not observable
pairwise_fastmath(::Type{Float32}) = true
pairwise_fastmath(::Type) = false

function Molly.tile_kernel_config(backend::CUDABackend, ::Type{T}, maxregs) where T
    kernel_backend = CUDABackend(; prefer_blocks=backend.prefer_blocks, always_inline=true,
                                 fastmath=pairwise_fastmath(T))
    options = maxregs === nothing ? (;) : (; maxregs=maxregs)
    return kernel_backend, options
end

# The kernels for which `Molly.kernel_fastmath` is `true` are compiled with fast math
Molly.fastmath_backend(b::CUDABackend) = CUDABackend(b.prefer_blocks, b.always_inline, true)

# A compiled kernel belongs to a context, which changes when the device is reset
Molly.kernel_cache_context(::CUDABackend) = CUDA.context()

# Static shared memory is limited to 48 KiB per block
Molly.max_tile_localmem(::CUDABackend) = 48 * 1024

## Concurrent forces

#=
The stream for the work launched by `Molly.run_side_work!`, with the events that order it
with the stream of the task. It has the highest priority, so that the blocks of its kernels
are run as soon as there is room on a multiprocessor rather than after the blocks of the
pairwise force kernel launched before them, which would leave little time to run at once.
Kept per task and device, as each task has its own stream.
=#
struct SideStream
    stream::CuStream
    started::CuEvent
    finished::CuEvent
end

function side_stream()
    key = (:molly_side_stream, CUDA.device())
    return get!(task_local_storage(), key) do
        SideStream(CuStream(; priority=last(CUDA.priority_range())),
                   CuEvent(CUDA.EVENT_DISABLE_TIMING), CuEvent(CUDA.EVENT_DISABLE_TIMING))
    end::SideStream
end

struct SideWork
    main::CuStream
    side::SideStream
end

function Molly.side_work_start(::CUDABackend)
    side = side_stream()
    main = CUDA.stream()
    CUDA.record(side.started, main)
    return SideWork(main, side)
end

#=
CUDA.jl makes the host wait for the stream that an array was last used on when it is used
on another, which here would wait for the pairwise force kernel. The events order the
streams instead, so the shared arrays are used without that and left marked as used on the
main stream. Other arrays used by the side work, such as those of the interactions, are
then marked as used on the side stream.
=#
function Molly.run_side_work!(f, w::SideWork, shared)
    s = w.side
    CUDA.wait(s.started, s.stream)
    synchronizing = map(unsynchronize!, shared)
    try
        CUDA.stream!(f, s.stream)
    finally
        foreach((x, sync) -> resynchronize!(x, sync, w.main), shared, synchronizing)
    end
    CUDA.record(s.finished, s.stream)
    return w
end

function Molly.side_work_finish!(w::SideWork)
    CUDA.wait(w.side.finished, w.main)
    return w
end

function unsynchronize!(x::CuArray)
    managed = x.data[]
    synchronizing = managed.synchronizing
    managed.synchronizing = false
    return synchronizing
end

unsynchronize!(x) = false

function resynchronize!(x::CuArray, synchronizing, stream)
    managed = x.data[]
    managed.synchronizing = synchronizing
    managed.stream = stream
    return x
end

resynchronize!(x, synchronizing, stream) = x

# The gradients of the forces are found without the side stream
Molly.concurrent_gpu_forces(sys::System{<:Any, <:CuArray}) = !sys.grad_safe

## Displacement check

# The last block of the check reduces the results of the blocks, which needs the
#   acquire-release atomics and device-scope fences that CUDA provides
Molly.fused_tile_check(::CUDABackend) = true

#=
The result of a check is written by the device to pinned host memory mapped into the
device address space, so that the host reads it without a copy once the event recorded
after the check has completed.
=#
function Molly.check_result_arrays(::CUDABackend, ::Type{T}, n::Integer) where T
    result = CuArray{T, 1, CUDACore.HostMemory}(undef, n)
    fill!(unsafe_wrap(Array, result), zero(T))
    return result, unsafe_wrap(Array, result), CuEvent(CUDACore.EVENT_DISABLE_TIMING)
end

function Molly.record_check!(::CUDABackend, event::CuEvent, scratch)
    CUDA.record(event, CUDA.stream())
    return scratch
end

#=
The host spins on the event rather than calling `CUDA.synchronize`, which after a short
spin hands the wait to another thread and can take hundreds of microseconds to wake up
once the event completes. That is the usual case when the device is behind the host, as
for the neighbor finders that build a list on the GPU, where it made the check cost about
15% of a step. After a second the normal synchronization is used.
=#
function Molly.wait_check_result!(::CUDABackend, event::CuEvent, scratch)
    if !CUDA.isdone(event)
        t_start = time_ns()
        while !CUDA.isdone(event)
            ccall(:jl_cpu_pause, Cvoid, ())
            # Lets other threads collect garbage while this one waits
            ccall(:jl_gc_safepoint, Cvoid, ())
            if time_ns() - t_start > 1_000_000_000
                CUDA.synchronize(event)
                break
            end
        end
    end
    return scratch.result_host
end

end
