# ANI-2x energy + forces timing: CPU (threaded) vs NVIDIA CUDA. The KA kernels are backend
# portable, so this is the CUDA counterpart of ani_gpu_compare.jl + ani_forces_gpu.jl (which
# target Apple Metal). Same analytic on-device path (compute_ani_energy_ka / compute_ani_forces_ka),
# just CuArray + CUDABackend() instead of MtlArray + MetalBackend().
#
#   julia --project=<env> -t8 benchmark/ani_cuda_compare.jl
# Env: ANI_SIZES (default "1000,2000,5000,8000,15954"). Writes results/ani_cuda_{energy,forces}.json.
# Needs a working CUDA GPU + `using CUDA` in the environment (add CUDA to the benchmark project).

# cuDNN is the MLDataDevices trigger that makes `Lux.gpu_device()` select CUDA (CUDA.jl alone is
# not enough; on Metal, Metal.jl is the trigger). Without it the NN params stay on the CPU.
using Molly, Lux, HDF5, KernelAbstractions, CUDA, cuDNN, StaticArrays, Unitful, LinearAlgebra
using JSON3
include(joinpath(@__DIR__, "ani_bench_common.jl"))

const REF = joinpath(@__DIR__, "..", "data", "ani_reference")
const PDB = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")
pot   = ANIPotential(joinpath(REF, "ani2x.h5"); ensemble_idx = 0)
n_sp  = length(pot.species_map)
valid = Set(keys(pot.species_map))
sizes = parse.(Int, split(get(ENV, "ANI_SIZES", "1000,2000,5000,8000,15954"), ","))

@assert CUDA.functional() "CUDA is not functional in this environment"
println("CUDA device: ", CUDA.name(CUDA.device()))

function load(nmax)
    coords = SVector{3,Float64}[]; elems = String[]
    open(PDB) do f
        for line in eachline(f)
            (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
            length(line) < 78 && continue
            e = strip(line[77:78]); e in valid || continue
            push!(coords, SVector(parse(Float64, line[31:38]), parse(Float64, line[39:46]),
                                  parse(Float64, line[47:54])))
            push!(elems, e); length(elems) == nmax && break
        end
    end
    coords, elems
end

# Synchronous GPU timing: block on the device so we measure real kernel time.
cubench(f; w=3, r=15) = (for _ in 1:w; f(); CUDA.synchronize(); end;
                         minimum(begin t=time(); f(); CUDA.synchronize(); time()-t end for _ in 1:r) * 1e3)

println(run_header(), " | CUDA ", CUDA.runtime_version())
println(rpad("N atoms", 9), rpad("E cpu-t8", 11), rpad("E cuda", 11), rpad("F cpu-t8", 11), "F cuda")

# NB: don't name these `energy`/`forces` — that would shadow Molly's `forces` function used below.
energy_res = Dict("cpu"=>Dict{String,Any}(), "cuda"=>Dict{String,Any}(), "env"=>run_env())
forces_res = Dict("cpu"=>Dict{String,Any}(), "cuda"=>Dict{String,Any}(), "env"=>run_env())

for N in sizes
    coords, elems = load(N)
    n  = length(coords); sp = Int32.([pot.species_map[e] for e in elems])
    nf = DistanceNeighborFinder(eligible=trues(n, n), dist_cutoff=(Float64(pot.cutoff)+1.0)u"Å")
    sys = System(atoms=[Atom(mass=1.0u"u") for _ in 1:n], coords=[c*u"Å" for c in coords],
        boundary=CubicBoundary(200.0u"Å"), atoms_data=[AtomData(element=e) for e in elems],
        general_inters=(ani=pot,), neighbor_finder=nf, force_units=u"eV/Å", energy_units=u"eV")
    nbrs = Molly.find_neighbors(sys)
    cu = CuArray([SVector{3,Float32}(c) for c in coords]); su = CuArray(sp)
    bdyf = CubicBoundary(200.0f0)

    # CPU column: Molly's standard API (matches the CPU numbers reported elsewhere).
    eC = bench(() -> potential_energy(sys)).min
    fC = bench(() -> forces(sys); repeats=3, samples=5).min
    # CUDA column: the on-device analytic path.
    eG = cubench(() -> CUDA.@sync Molly.compute_ani_energy_ka(cu, su, pot, n_sp; backend=CUDABackend(), neighbors=nbrs, boundary=bdyf))
    fG = cubench(() -> CUDA.@sync Molly.compute_ani_forces_ka(cu, su, pot, n_sp; backend=CUDABackend(), neighbors=nbrs, boundary=bdyf))

    energy_res["cpu"][string(n)] = Dict("min"=>eC); energy_res["cuda"][string(n)] = Dict("min"=>eG)
    forces_res["cpu"][string(n)] = Dict("min"=>fC); forces_res["cuda"][string(n)] = Dict("min"=>fG)
    println(rpad(n,9), rpad(round(eC,digits=1),11), rpad(round(eG,digits=1),11),
            rpad(round(fC,digits=1),11), round(fG,digits=1))
end

write_json(joinpath(@__DIR__, "results", "ani_cuda_energy.json"), energy_res)
write_json(joinpath(@__DIR__, "results", "ani_cuda_forces.json"), forces_res)
println("done.")
