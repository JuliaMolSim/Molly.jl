# ANI forces timing: CPU vs Apple Metal, both via the single analytic path
# (`AtomsCalculators.forces!` → `compute_ani_forces_ka`: forward AEV → NN VJP → backward
# radial/angular kernels → F = -∂E/∂r). The KA CPU backend and Metal run the same code; only the
# device differs. Both consume the finder's NeighborList. Counterpart of ani_gpu_compare.jl (energy).
#
#   julia --project=<env> -t8 benchmark/ani_forces_gpu.jl
# Env: ANI_FSIZES (Metal sizes, comma list), ANI_CPU_SIZES (CPU sizes), ANI_ENSEMBLE (0|full).
# Writes benchmark/results/ani_forces.json.

using Molly, Lux, HDF5, KernelAbstractions, Metal, StaticArrays, Unitful, LinearAlgebra
using JSON3
include(joinpath(@__DIR__, "ani_bench_common.jl"))

const REF = joinpath(@__DIR__, "..", "data", "ani_reference")
ens_full  = get(ENV, "ANI_ENSEMBLE", "0") == "full"
pot   = ens_full ? ANIPotential(joinpath(REF, "ani2x.h5")) : ANIPotential(joinpath(REF, "ani2x.h5"); ensemble_idx=0)
n_sp  = length(pot.species_map)
valid = Set(keys(pot.species_map))

msizes = parse.(Int, split(get(ENV, "ANI_FSIZES", "200,500,1000,2000,4000,8000,15954"), ","))
csizes = parse.(Int, split(get(ENV, "ANI_CPU_SIZES", "200,500,1000,2000,4000,8000,15954"), ","))

function load(nmax)
    coords = SVector{3,Float64}[]; elems = String[]
    open(joinpath(REF, "..", "6mrr_equil.pdb")) do f
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

build_sys(coords, elems) = System(
    atoms = [Atom(mass=1.0u"u") for _ in eachindex(coords)],
    coords = [c*u"Å" for c in coords], boundary = CubicBoundary(200.0u"Å"),
    atoms_data = [AtomData(element=e) for e in elems],
    general_inters = (ani=pot,),
    neighbor_finder = DistanceNeighborFinder(eligible=trues(length(coords), length(coords)),
                                             dist_cutoff=(Float64(pot.cutoff)+1.0)u"Å"),
    force_units = u"eV/Å", energy_units = u"eV")

println("ANI forces timing | ensemble=", ens_full ? "8-member" : "single",
        " | Metal functional: ", Metal.functional())
println(run_header())
println(rpad("N atoms", 10), rpad("CPU analytic (ms)", 18), "Metal (ms)")

results = Dict{String,Any}("env" => run_env(), "ensemble" => ens_full ? "full" : "single",
                           "cpu" => Dict{String,Any}(), "metal" => Dict{String,Any}())

allsizes = sort(unique(vcat(msizes, csizes)))
for n in allsizes
    coords, elems = load(n)
    nn  = length(coords)
    sys = build_sys(coords, elems)
    nbrs = Molly.find_neighbors(sys)

    # CPU analytic forces via the single path (forces(sys) → compute_ani_forces_ka, KA CPU backend).
    t_cpu = NaN
    if n in csizes
        r = bench(() -> forces(sys); repeats=3, samples=5, seconds=60.0)
        t_cpu = r.min
        results["cpu"][string(nn)] = Dict("min"=>r.min, "median"=>r.median, "iqr"=>r.iqr)
    end

    # Metal on-device analytic forces.
    t_mtl = NaN
    if n in msizes
        sp   = Int32.([pot.species_map[e] for e in elems])
        cM   = MtlArray([SVector{3,Float32}(c) for c in coords])
        sM   = MtlArray(sp)
        bdy  = CubicBoundary(200.0f0)
        r = bench(() -> Molly.compute_ani_forces_ka(cM, sM, pot, n_sp; backend=MetalBackend(),
                                              neighbors=nbrs, boundary=bdy); repeats=5, samples=15)
        t_mtl = r.min
        results["metal"][string(nn)] = Dict("min"=>r.min, "median"=>r.median, "iqr"=>r.iqr)
    end

    println(rpad(nn, 10), rpad(isnan(t_cpu) ? "-" : round(t_cpu, digits=1), 18),
            isnan(t_mtl) ? "-" : round(t_mtl, digits=2))
end

write_json(joinpath(@__DIR__, "results", "ani_forces.json"), results)
