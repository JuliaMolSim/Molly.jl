# Shared benchmarking harness for the ANI scripts: timing with repeats + a variance band,
# a run/version header, and JSON output. `include` this from the individual scripts.
#
#   using JSON3           # optional; JSON writing is skipped if it is not loaded
#   include("ani_bench_common.jl")
#   r = bench(() -> compute_ani_energy_ka(...))   # → (min, median, iqr, bytes) in ms
#
# Timing is min-of-`samples` per repeat (best-case, low-noise) with `repeats` repeats, and the
# stats are taken across the per-repeat minima so `iqr` is an honest laptop run-to-run band.

using Statistics, LinearAlgebra

"""
    bench(f; repeats=5, samples=10, seconds=10.0) -> (; min, median, iqr, bytes, repeats)

Warm `f` once (compile), then take the min wall time (ms) of up to `samples` calls per repeat,
across `repeats` repeats. `min`/`median`/`iqr` are over the per-repeat minima; `bytes` is the
last measured allocation. For GPU work `f` must be synchronous (e.g. return a host value).
"""
function bench(f; repeats=5, samples=10, seconds=10.0)
    f()                                    # warm / compile
    mins  = Float64[]
    bytes = 0
    for _ in 1:repeats
        best = Inf
        t0 = time(); s = 0
        while s < samples && (time() - t0) < seconds
            st   = @timed f()
            best = min(best, st.time * 1e3) # s → ms
            bytes = st.bytes
            s += 1
        end
        push!(mins, best)
    end
    q1 = quantile(mins, 0.25); q3 = quantile(mins, 0.75)
    (min = minimum(mins), median = median(mins), iqr = q3 - q1, bytes = bytes, repeats = repeats)
end

"Human-readable run header (Julia + CPU + thread/BLAS config)."
function run_header(; threads = Threads.nthreads())
    string("Julia ", VERSION, " | ", Sys.CPU_NAME, " | threads=", threads,
           " | BLAS=", LinearAlgebra.BLAS.get_num_threads())
end

"Machine-readable environment dict for the JSON header (adds Metal status if `Metal` is loaded)."
function run_env(; threads = Threads.nthreads())
    env = Dict{String,Any}(
        "julia"   => string(VERSION),
        "cpu"     => Sys.CPU_NAME,
        "threads" => threads,
        "blas"    => LinearAlgebra.BLAS.get_num_threads(),
    )
    if isdefined(Main, :Metal)
        try; env["metal_functional"] = Main.Metal.functional(); catch; end
    end
    env
end

"Write `obj` to `path` as pretty JSON (no-op with a warning if JSON3 is not loaded)."
function write_json(path, obj)
    if !isdefined(Main, :JSON3)
        @warn "JSON3 not loaded — skipping $path (add `using JSON3`)"
        return
    end
    mkpath(dirname(path))
    open(path, "w") do io
        Main.JSON3.pretty(io, obj)
    end
    println("wrote ", path)
end
