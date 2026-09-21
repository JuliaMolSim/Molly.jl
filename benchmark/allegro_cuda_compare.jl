# Allegro energy: CPU vs NVIDIA CUDA (run on the RTX 5080 box). Verifies the GPU energy matches
# the CPU forward, then times both across system sizes and writes results/allegro_cuda_energy.json.
#
#   julia --project=<env-with-Molly+HDF5+CUDA+JSON3> benchmark/allegro_cuda_compare.jl
# Env: ALLEGRO_SIZES (default "64,128,256,512,1024"), ALLEGRO_SPACING (Å, default "2.5").

using Molly, HDF5, CUDA, JSON3, Random, Printf
using Molly: SVector, to_device
const KA = Molly.KernelAbstractions

include(joinpath(@__DIR__, "allegro_bench_common.jl"))

const H5  = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")
const RES = joinpath(@__DIR__, "results")
_sizes()   = parse.(Int, split(get(ENV, "ALLEGRO_SIZES", "64,128,256,512,1024"), ","))
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
    if !CUDA.functional()
        @warn "CUDA not functional — aborting"; return
    end
    println("CUDA device: ", CUDA.name(CUDA.device()))
    pot = AllegroPotential(H5; T=Float64)
    m = pot.model; rc = m.r_c
    gpu = Molly.build_allegro_gpu(m, CUDABackend(), Float64)
    sizes = _sizes()
    println("Allegro CUDA vs CPU | ", run_header())
    println("model: C=$(m.C) H=$(m.H) layers=$(m.L) r_c=$(rc)  sizes=$(sizes)\n")

    E_res = Dict{String,Any}("cpu" => Dict{String,Any}(), "cuda" => Dict{String,Any}())
    @printf("| %6s | %8s | %12s | %12s | %10s | %10s |\n",
            "atoms", "edges", "CPU (ms)", "CUDA (ms)", "speedup", "reldiff")
    @printf("|%s|%s|%s|%s|%s|%s|\n", "-"^8, "-"^10, "-"^14, "-"^14, "-"^12, "-"^12)
    for n in sizes
        coords, species = make_system(n)
        n_edges = sum(length, Molly.neighbour_lists(coords, nothing, rc))
        E_cpu = Molly.allegro_total_energy(m, coords, species, nothing, rc)
        cg = to_device(coords, CuArray)
        E_gpu = Molly.compute_allegro_energy_ka(m, cg, species, nothing;
                    backend=CUDABackend(), T=Float64, gpu=gpu)
        reld = abs(E_gpu - E_cpu) / max(abs(E_cpu), 1e-12)
        ec = bench(() -> Molly.allegro_total_energy(m, coords, species, nothing, rc))
        eg = bench(() -> (e = Molly.compute_allegro_energy_ka(m, cg, species, nothing;
                            backend=CUDABackend(), T=Float64, gpu=gpu); CUDA.synchronize(); e))
        E_res["cpu"][string(n)]  = Dict("min"=>ec.min, "edges"=>n_edges)
        E_res["cuda"][string(n)] = Dict("min"=>eg.min, "edges"=>n_edges)
        @printf("| %6d | %8d | %12.3f | %12.3f | %9.1f× | %10.2e |\n",
                n, n_edges, ec.min, eg.min, ec.min/eg.min, reld)
    end
    write_json(joinpath(RES, "allegro_cuda_energy.json"), merge(Dict("header"=>run_env()), E_res))
end

main()
