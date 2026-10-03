# Allegro FORCES: CPU vs Apple Metal (run on Apple Silicon). Verifies the GPU-portable analytic
# forces match the CPU analytic backward, then times CPU (tagged by thread count) and Metal across
# sizes, merging into results/allegro_metal_forces.json. Run once per thread count for the CPU
# t1/t8 series:
#
#   julia --project=<env-with-Molly+HDF5+Metal+JSON3> -t8 benchmark/allegro_forces_metal.jl
#   julia --project=<env-with-Molly+HDF5+Metal+JSON3> -t1 benchmark/allegro_forces_metal.jl
# Env: ALLEGRO_SIZES (default "64,128,256,512,1024,2048,4096"), ALLEGRO_CPU_MAX (largest N still
# timed on the CPU, default "2048" — the CPU analytic backward is O(N) but heavy), ALLEGRO_SPACING.

using Molly, HDF5, Metal, JSON3, Random, Printf
using Molly: SVector, to_device

include(joinpath(@__DIR__, "allegro_bench_common.jl"))

const H5  = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")
const RES = joinpath(@__DIR__, "results")
const OUT = joinpath(RES, "allegro_metal_forces.json")
_sizes()   = parse.(Int, split(get(ENV, "ALLEGRO_SIZES", "64,128,256,512,1024,2048,4096"), ","))
_cpu_max() = parse(Int, get(ENV, "ALLEGRO_CPU_MAX", "2048"))
_spacing() = parse(Float64, get(ENV, "ALLEGRO_SPACING", "2.5"))

function make_system(n; a=_spacing(), jitter=0.2, S=2, seed=1)
    rng = MersenneTwister(seed); side = ceil(Int, cbrt(n)); coords = SVector{3,Float64}[]
    for x in 0:side-1, y in 0:side-1, z in 0:side-1
        length(coords) == n && break
        push!(coords, SVector{3,Float64}(a*x, a*y, a*z) .+
              jitter .* (2 .* SVector{3,Float64}(rand(rng), rand(rng), rand(rng)) .- 1))
    end
    return coords, rand(rng, 1:S, n)
end

maxdiff(Fh, Fcpu) = maximum(maximum(abs.(Float64.(Fh[:, i]) .- Fcpu[i])) for i in eachindex(Fcpu))

function main()
    metal_ok = Metal.functional()
    pot = AllegroPotential(H5; T=Float64)
    m = pot.model; rc = m.r_c
    gpu = metal_ok ? Molly.build_allegro_gpu(m, Metal.MetalBackend(), Float32) : nothing
    sizes = _sizes(); cpu_max = _cpu_max()
    cpu_key = "cpu_t$(Threads.nthreads())"
    println("Allegro FORCES Metal vs CPU | ", run_header(), " | Metal functional=", metal_ok)
    println("model: C=$(m.C) H=$(m.H) layers=$(m.L) r_c=$(rc)  sizes=$(sizes)  cpu_max=$cpu_max  (key $cpu_key)\n")

    prev = isfile(OUT) ? JSON3.read(read(OUT, String)) : nothing
    res = Dict{String,Any}(cpu_key => Dict{String,Any}(), "metal" => Dict{String,Any}())
    if !isnothing(prev)
        for k in keys(prev)
            k == :header && continue
            res[string(k)] = Dict{String,Any}(string(kk) => Dict(string(kkk)=>vvv for (kkk,vvv) in vv)
                                               for (kk, vv) in prev[k])
        end
        res[cpu_key] = Dict{String,Any}()
    end

    @printf("| %6s | %8s | %12s | %12s | %10s | %10s |\n",
            "atoms", "edges", "CPU (ms)", "Metal (ms)", "speedup", "maxΔF")
    @printf("|%s|%s|%s|%s|%s|%s|\n", "-"^8, "-"^10, "-"^14, "-"^14, "-"^12, "-"^12)
    for n in sizes
        coords, species = make_system(n)
        n_edges = sum(length, Molly.neighbour_lists(coords, nothing, rc))
        _, F_cpu = Molly.allegro_energy_and_forces(m, coords, species, nothing, rc)
        cpu_str = "—"
        if n <= cpu_max
            fc = bench(() -> Molly.allegro_energy_and_forces(m, coords, species, nothing, rc))
            res[cpu_key][string(n)] = Dict("min"=>fc.min, "edges"=>n_edges)
            cpu_str = @sprintf("%.3f", fc.min)
        end
        mt_str = "—"; sp = "—"; mdf = NaN
        if metal_ok
            coords32 = [SVector{3,Float32}(Float32.(c)...) for c in coords]
            cg = to_device(coords32, MtlArray)
            _, Fdev = Molly.compute_allegro_forces_ka(m, cg, species, nothing;
                          backend=Metal.MetalBackend(), T=Float32, gpu=gpu)
            Metal.synchronize()
            mdf = maxdiff(Array(Fdev), F_cpu)
            fg = bench(() -> (r = Molly.compute_allegro_forces_ka(m, cg, species, nothing;
                                  backend=Metal.MetalBackend(), T=Float32, gpu=gpu);
                              Metal.synchronize(); r))
            res["metal"][string(n)] = Dict("min"=>fg.min, "edges"=>n_edges)
            mt_str = @sprintf("%.3f", fg.min)
            n <= cpu_max && (sp = @sprintf("%.1f×", res[cpu_key][string(n)]["min"]/fg.min))
        end
        @printf("| %6d | %8d | %12s | %12s | %10s | %10.2e |\n", n, n_edges, cpu_str, mt_str, sp, mdf)
    end
    write_json(OUT, merge(Dict("header"=>run_env()), res))
    println("\nwrote ", OUT)
end

main()
