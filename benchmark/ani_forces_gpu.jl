# ANI forces timing: CPU (Enzyme reverse-mode AD, the production CPU path) vs Apple Metal
# (on-device analytic backward, `compute_ani_forces_ka`). This is the GPU-forces counterpart of
# ani_gpu_compare.jl (which does energy). Metal times the fully on-device path: GPU AEV → NN VJP
# → backward radial/angular kernels → F = -∂E/∂r. Both consume the finder's NeighborList.
#
#   julia --project=<env> -t8 benchmark/ani_forces_gpu.jl
# Env: ANI_FSIZES (Metal sizes, comma list), ANI_ENZYME_SIZES (CPU-Enzyme sizes; slow),
#      ANI_ENSEMBLE (0|full). Writes benchmark/results/ani_forces.json.

using Molly, Lux, HDF5, KernelAbstractions, Enzyme, Metal, StaticArrays, Unitful, LinearAlgebra
using JSON3
include(joinpath(@__DIR__, "ani_bench_common.jl"))

const REF = joinpath(@__DIR__, "..", "data", "ani_reference")
ens_full  = get(ENV, "ANI_ENSEMBLE", "0") == "full"
pot   = ens_full ? ANIPotential(joinpath(REF, "ani2x.h5")) : ANIPotential(joinpath(REF, "ani2x.h5"); ensemble_idx=0)
n_sp  = length(pot.species_map)
valid = Set(keys(pot.species_map))

msizes = parse.(Int, split(get(ENV, "ANI_FSIZES", "200,500,1000,2000,4000,8000,15954"), ","))
esizes = parse.(Int, split(get(ENV, "ANI_ENZYME_SIZES", "200,500,1000,2000"), ","))

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
println(rpad("N atoms", 10), rpad("CPU-Enzyme (ms)", 18), "Metal (ms)")

results = Dict{String,Any}("env" => run_env(), "ensemble" => ens_full ? "full" : "single",
                           "cpu_enzyme" => Dict{String,Any}(), "metal" => Dict{String,Any}())

allsizes = sort(unique(vcat(msizes, esizes)))
for n in allsizes
    coords, elems = load(n)
    nn  = length(coords)
    sys = build_sys(coords, elems)
    nbrs = Molly.find_neighbors(sys)

    # CPU Enzyme forces (production path) — slow, so fewer samples and only the requested sizes.
    t_cpu = NaN
    if n in esizes
        r = bench(() -> forces(sys); repeats=3, samples=3, seconds=60.0)
        t_cpu = r.min
        results["cpu_enzyme"][string(nn)] = Dict("min"=>r.min, "median"=>r.median, "iqr"=>r.iqr)
    end

    # Metal on-device analytic forces.
    t_mtl = NaN
    if n in msizes
        sp   = Int32.([pot.species_map[e] for e in elems])
        cM   = MtlArray([SVector{3,Float32}(c) for c in coords])
        sM   = MtlArray(sp)
        bdy  = CubicBoundary(200.0f0)
        r = bench(() -> compute_ani_forces_ka(cM, sM, pot, n_sp; backend=MetalBackend(),
                                              neighbors=nbrs, boundary=bdy); repeats=5, samples=15)
        t_mtl = r.min
        results["metal"][string(nn)] = Dict("min"=>r.min, "median"=>r.median, "iqr"=>r.iqr)
    end

    println(rpad(nn, 10), rpad(isnan(t_cpu) ? "-" : round(t_cpu, digits=1), 18),
            isnan(t_mtl) ? "-" : round(t_mtl, digits=2))
end

write_json(joinpath(@__DIR__, "results", "ani_forces.json"), results)
