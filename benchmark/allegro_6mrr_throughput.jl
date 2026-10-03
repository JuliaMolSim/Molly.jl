# 6mrr trajectory-throughput head-to-head, Molly side: time a single Allegro energy+forces evaluation
# on the full 6mrr system (15,954 atoms) for one Molly backend and record the MD throughput it implies
# (ns/day at dt = 0.1 fs). Pairs with benchmark/allegro_6mrr_throughput.py (nequip + allegro-jax);
# together they build results/allegro_traj_throughput.json for the trajectory figure.
#
#   ALLEGRO_TRAJ_BACKEND=cpu  julia --project=<env> -t8 benchmark/allegro_6mrr_throughput.jl
#   ALLEGRO_TRAJ_BACKEND=cuda julia --project=<env>     benchmark/allegro_6mrr_throughput.jl
#   ALLEGRO_TRAJ_BACKEND=metal julia --project=<env>    benchmark/allegro_6mrr_throughput.jl
using Molly, HDF5, JSON3, Printf
using Molly: SVector, to_device
const BK = lowercase(get(ENV, "ALLEGRO_TRAJ_BACKEND", "cpu"))
if BK == "cuda"
    using CUDA
    devc(x) = to_device(x, CuArray); sync() = CUDA.synchronize(); backend() = CUDABackend(); const TT = Float64
elseif BK == "metal"
    using Metal
    devc(x) = to_device(x, MtlArray); sync() = Metal.synchronize(); backend() = Metal.MetalBackend(); const TT = Float32
else
    devc(x) = x; sync() = nothing; backend() = nothing; const TT = Float64
end

const ROOT = dirname(@__DIR__)
const H5   = joinpath(ROOT, "data", "allegro_reference", "allegro_model.h5")
const PDB  = get(ENV, "ALLEGRO_PDB", joinpath(ROOT, "data", "6mrr_equil.pdb"))
const RES  = joinpath(@__DIR__, "results")
const ELEM_SPECIES = Dict("H"=>1, "C"=>2, "N"=>2, "O"=>2, "S"=>2)

function parse_pdb(path)
    coords = SVector{3,Float64}[]; elems = String[]; box = SVector{3,Float64}(0,0,0)
    for ln in eachline(path)
        if startswith(ln, "CRYST1")
            box = SVector{3,Float64}(parse(Float64, ln[7:15]), parse(Float64, ln[16:24]), parse(Float64, ln[25:33]))
        elseif startswith(ln, "ATOM") || startswith(ln, "HETATM")
            push!(coords, SVector{3,Float64}(parse(Float64, ln[31:38]), parse(Float64, ln[39:46]), parse(Float64, ln[47:54])))
            push!(elems, strip(ln[77:78]))
        end
    end
    coords, elems, box
end

coords, elems, box = parse_pdb(PDB)
n = length(coords); species = [ELEM_SPECIES[e] for e in elems]
pot = AllegroPotential(H5; T=Float64); m = pot.model; rc = m.r_c
boundary = CubicBoundary(box[1], box[2], box[3])
key = BK == "cpu" ? "cpu_t$(Threads.nthreads())" : BK
println("6mrr throughput | backend=$BK key=$key | atoms=$n")

function timeit(f; reps=3)
    f(); best = Inf
    for _ in 1:reps
        t = time(); f(); sync(); best = min(best, (time() - t) * 1e3)
    end
    best
end

ms = if BK == "cpu"
    timeit(() -> Molly.allegro_energy_and_forces(m, coords, species, boundary, rc))
else
    gpu = Molly.build_allegro_gpu(m, backend(), TT)
    cdev = devc([SVector{3,TT}(TT.(c)...) for c in coords])
    timeit(() -> Molly.compute_allegro_forces_ka(m, cdev, species, boundary; backend=backend(), T=TT, gpu=gpu))
end
nsday = 8.64 / ms   # dt = 0.1 fs
@printf("6mrr: %.1f ms/step  ->  %.4f ns/day\n", ms, nsday)

mkpath(RES)
path = joinpath(RES, "allegro_traj_throughput.json")
prev = isfile(path) ? JSON3.read(read(path, String), Dict{String,Dict{String,Any}}) : Dict{String,Dict{String,Any}}()
prev["molly_$key"] = Dict{String,Any}("atoms"=>n, "ms_step"=>ms, "ns_day"=>nsday, "status"=>"ok")
open(path, "w") do io; JSON3.pretty(io, prev); end
println("wrote ", path)
