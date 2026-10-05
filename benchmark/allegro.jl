# Benchmark the native bit-exact Allegro potential (AllegroPackageModel): wall time for one
# energy + analytic-forces evaluation vs system size, on CPU and (optionally) a GPU backend. The
# model is the real nequip-allegro architecture (l_max=2, 2 layers, 32 scalar / 8 tensor features),
# loaded from the package's exported weights; timings are representative of that architecture (they
# depend on its shape, not the weight values). Pairs with benchmark/allegro_package_bench.py, which
# times the real package on the same config, for benchmark/allegro_benchmarks.md.
#
#   ALLEGRO_BK=cpu   julia --project=<env>  benchmark/allegro.jl
#   ALLEGRO_BK=metal julia --project=<env>  benchmark/allegro.jl
#   ALLEGRO_BK=cuda  julia --project=<env>  benchmark/allegro.jl
using Molly, HDF5, JSON3, Random, Printf
using Molly: SVector, to_device, allegro_package_total_energy, allegro_package_energy_and_forces,
             compute_allegro_package_energy_ka, compute_allegro_package_energy_and_forces_ka,
             build_allegro_package_gpu, pkg_build_edges, pkg_edges_cpu

const BK  = lowercase(get(ENV, "ALLEGRO_BK", "cpu"))
const ROOT = dirname(@__DIR__)
const H5   = joinpath(ROOT, "data", "allegro_reference", "allegro_package_weights.h5")
const RES  = joinpath(@__DIR__, "results")
const SIZES = [parse(Int, s) for s in split(get(ENV, "ALLEGRO_SIZES", "100,250,500,1000,2000"), ",")]
const DENSITY = 0.09   # atoms / Å³ (condensed-phase-ish), sets the box so avg neighbours is realistic

if BK == "cuda"
    using CUDA
    const TT = Float64; backend() = CUDABackend(); devc(x) = to_device(x, CuArray); sync() = CUDA.synchronize()
elseif BK == "metal"
    using Metal
    const TT = Float32; backend() = Metal.MetalBackend(); devc(x) = to_device(x, MtlArray); sync() = Metal.synchronize()
else
    const TT = Float64; backend() = nothing; devc(x) = x; sync() = nothing
end

function random_system(n, rng)
    L = cbrt(n / DENSITY)                          # cubic box side (Å)
    coords = [SVector{3,Float64}(L*rand(rng), L*rand(rng), L*rand(rng)) for _ in 1:n]
    species = [rand(rng, 0:3) for _ in 1:n]        # 0-based H/C/N/O
    return coords, species, L
end

function timeit(f; reps=4)
    f(); best = Inf
    for _ in 1:reps
        t = time(); f(); sync(); best = min(best, (time()-t)*1e3)
    end
    best
end

m = load_allegro_package(H5; T=Float64)
rng = MersenneTwister(1)
key = BK == "cpu" ? "cpu_t$(Threads.nthreads())" : BK
println("native Allegro benchmark | backend=$BK T=$TT | sizes=$SIZES")
gpu = BK == "cpu" ? nothing : build_allegro_package_gpu(m, backend(), TT)

rows = Dict{String,Any}()
for n in SIZES
    coords, species, L = random_system(n, rng)
    ms_e, ms_ef = if BK == "cpu"
        ced = get(ENV, "ALLEGRO_INCLUDE_NL", "0") == "1" ? nothing : pkg_edges_cpu(coords, m.r_max)
        (timeit(() -> allegro_package_total_energy(m, coords, species; edges=ced)),
         timeit(() -> allegro_package_energy_and_forces(m, coords, species; edges=ced)))
    else
        cdev = devc([SVector{3,TT}(TT.(c)...) for c in coords])
        # Precompute the neighbour list once (reused across MD steps in practice), so the timed region
        # is the model evaluation — matching the nequip-allegro / allegro-jax benches, which also pass
        # precomputed edges. ALLEGRO_INCLUDE_NL=1 instead times the full build+evaluate pipeline.
        ed = get(ENV, "ALLEGRO_INCLUDE_NL", "0") == "1" ? nothing :
             pkg_build_edges(cdev, TT(gpu.r_max), zero(TT), zero(TT), zero(TT); backend=backend())
        (timeit(() -> compute_allegro_package_energy_ka(m, cdev, species;
                        backend=backend(), T=TT, gpu=gpu, edges=ed)),
         timeit(() -> compute_allegro_package_energy_and_forces_ka(m, cdev, species;
                        backend=backend(), T=TT, gpu=gpu, edges=ed)))
    end
    rows["n$n"] = Dict("atoms"=>n, "box_A"=>L, "ms_energy"=>ms_e, "ms_energy_forces"=>ms_ef)
    @printf("  N=%5d  box=%.1f Å   energy %8.2f ms   energy+forces %8.2f ms\n", n, L, ms_e, ms_ef)
end

mkpath(RES)
path = joinpath(RES, "allegro_bench.json")
prev = isfile(path) ? JSON3.read(read(path, String), Dict{String,Any}) : Dict{String,Any}()
prev["native_$key"] = rows
open(path, "w") do io; JSON3.pretty(io, prev); end
println("wrote ", path)
