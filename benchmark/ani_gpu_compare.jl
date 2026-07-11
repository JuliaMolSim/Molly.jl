# Molly ANI energy timing on CPU and Apple Metal, for comparison against TorchANI.
#
# Run:  julia --project=<env> -t 8 benchmark/ani_gpu_compare.jl
# Env:  ANI_SIZES  comma list of atom counts (default "500,1000,2000")
#
# Produces a per-size table of Molly energy time (ms) on CPU and on Metal, so the numbers
# line up with test/torchani_reference.py --device {cpu,mps}. Metal times the on-device
# path compute_ani_energy_ka (GPU AEV + on-device NN). Molly GPU forces don't exist yet.

using Molly, Lux, HDF5, KernelAbstractions, Metal, BenchmarkTools, StaticArrays, Unitful, LinearAlgebra

const H5_PATH  = joinpath(@__DIR__, "..", "data", "ani_reference", "ani2x.h5")
const PDB_PATH = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")
const SIZES    = parse.(Int, split(get(ENV, "ANI_SIZES", "500,1000,2000"), ","))

pot   = ANIPotential(H5_PATH; ensemble_idx=0)
p     = pot.aev_params
n_sp  = length(pot.species_map)
valid = Set(keys(pot.species_map))

function load(n_max)
    coords = SVector{3,Float64}[]; elems = String[]
    open(PDB_PATH) do f
        for line in eachline(f)
            (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
            length(line) < 78 && continue
            e = strip(line[77:78]); e in valid || continue
            push!(coords, SVector(parse(Float64, line[31:38]), parse(Float64, line[39:46]),
                                  parse(Float64, line[47:54])))
            push!(elems, e); length(elems) == n_max && break
        end
    end
    coords, elems
end

# min wall time (ms) over a warm run
function bench_ms(f; samples=20, seconds=15.0)
    f()
    b = @benchmark $f() samples=samples seconds=seconds evals=1
    minimum(b).time / 1e6
end

println("Molly ANI energy timing (single ensemble member) | Metal functional: ", Metal.functional())
println(rpad("N atoms", 10), rpad("CPU energy (ms)", 18), "Metal energy (ms)")

for n in SIZES
    coords, elems = load(n)
    nn = length(coords)
    species = [pot.species_map[e] for e in elems]

    # CPU: potential_energy through the full path (buffered AEV + batched NN), with a finder.
    nf = DistanceNeighborFinder(eligible=trues(nn,nn), dist_cutoff=(Float64(pot.cutoff)+1.0)u"Å")
    sys = System(atoms=[Atom(mass=1.0u"u") for _ in 1:nn], coords=[c*u"Å" for c in coords],
        boundary=CubicBoundary(200.0u"Å"), atoms_data=[AtomData(element=e) for e in elems],
        general_inters=(ani=pot,), neighbor_finder=nf, force_units=u"eV/Å", energy_units=u"eV")
    t_cpu = bench_ms(() -> potential_energy(sys))

    # Metal: on-device AEV + NN via compute_ani_energy_ka, consuming the finder NeighborList.
    nbrs   = Molly.find_neighbors(sys)
    c32    = MtlArray([SVector{3,Float32}(c) for c in coords])
    s32    = MtlArray(Int32.(species))
    bdy32  = CubicBoundary(200.0f0)
    t_mtl  = bench_ms(() -> Molly.compute_ani_energy_ka(c32, s32, pot, n_sp;
                          backend=MetalBackend(), neighbors=nbrs, boundary=bdy32))

    println(rpad(nn, 10), rpad(round(t_cpu, digits=2), 18), round(t_mtl, digits=2))
end
