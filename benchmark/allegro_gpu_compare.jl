# Allegro energy: CPU vs Apple Metal (run on Apple Silicon). Verifies the GPU energy matches the
# CPU forward, then times CPU (tagged by thread count) and Metal across sizes, merging into
# results/allegro_metal_energy.json. Run once per thread count for the CPU t1/t8 series:
#
#   julia --project=<env-with-Molly+HDF5+Metal+JSON3> -t8 benchmark/allegro_gpu_compare.jl
#   julia --project=<env-with-Molly+HDF5+Metal+JSON3> -t1 benchmark/allegro_gpu_compare.jl
# Env: ALLEGRO_SIZES (default "64,128,256,512,1024,2048"), ALLEGRO_SPACING (Å, default "2.5").

using Molly, HDF5, Metal, JSON3, Random, Printf
using Molly: SVector, to_device
const KA = Molly.KernelAbstractions

include(joinpath(@__DIR__, "allegro_bench_common.jl"))

const H5  = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")
const RES = joinpath(@__DIR__, "results")
const OUT = joinpath(RES, "allegro_metal_energy.json")
_sizes()   = parse.(Int, split(get(ENV, "ALLEGRO_SIZES", "64,128,256,512,1024,2048"), ","))
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

function main()
    metal_ok = Metal.functional()
    pot = AllegroPotential(H5; T=Float64)
    m = pot.model; rc = m.r_c
    gpu = metal_ok ? Molly.build_allegro_gpu(m, Metal.MetalBackend(), Float32) : nothing
    sizes = _sizes()
    cpu_key = "cpu_t$(Threads.nthreads())"
    println("Allegro Metal vs CPU | ", run_header(), " | Metal functional=", metal_ok)
    println("model: C=$(m.C) H=$(m.H) layers=$(m.L) r_c=$(rc)  sizes=$(sizes)  (key $cpu_key)\n")

    prev = isfile(OUT) ? JSON3.read(read(OUT, String)) : nothing
    E_res = Dict{String,Any}(cpu_key => Dict{String,Any}(), "metal" => Dict{String,Any}())
    if !isnothing(prev)
        for k in keys(prev)
            k == :header && continue
            E_res[string(k)] = Dict{String,Any}(string(kk) => Dict(string(kkk)=>vvv for (kkk,vvv) in vv)
                                                 for (kk, vv) in prev[k])
        end
        E_res[cpu_key] = Dict{String,Any}()
    end

    @printf("| %6s | %8s | %12s | %12s | %10s | %10s |\n",
            "atoms", "edges", "CPU (ms)", "Metal (ms)", "speedup", "reldiff")
    @printf("|%s|%s|%s|%s|%s|%s|\n", "-"^8, "-"^10, "-"^14, "-"^14, "-"^12, "-"^12)
    for n in sizes
        coords, species = make_system(n)
        n_edges = sum(length, Molly.neighbour_lists(coords, nothing, rc))
        E_cpu = Molly.allegro_total_energy(m, coords, species, nothing, rc)
        ec = bench(() -> Molly.allegro_total_energy(m, coords, species, nothing, rc))
        E_res[cpu_key][string(n)] = Dict("min"=>ec.min, "edges"=>n_edges)
        mt_str = "—"; sp = "—"; reld = NaN
        if metal_ok
            coords32 = [SVector{3,Float32}(Float32.(c)...) for c in coords]
            cg = to_device(coords32, MtlArray)
            E_gpu = Molly.compute_allegro_energy_ka(m, cg, species, nothing;
                        backend=Metal.MetalBackend(), T=Float32, gpu=gpu)
            reld = abs(E_gpu - E_cpu) / max(abs(E_cpu), 1f-6)
            eg = bench(() -> (e = Molly.compute_allegro_energy_ka(m, cg, species, nothing;
                                backend=Metal.MetalBackend(), T=Float32, gpu=gpu);
                              Metal.synchronize(); e))
            E_res["metal"][string(n)] = Dict("min"=>eg.min, "edges"=>n_edges)
            mt_str = @sprintf("%.3f", eg.min); sp = @sprintf("%.1f×", ec.min/eg.min)
        end
        @printf("| %6d | %8d | %12.3f | %12s | %10s | %10.2e |\n", n, n_edges, ec.min, mt_str, sp, reld)
    end
    write_json(OUT, merge(Dict("header"=>run_env()), E_res))
end

main()
