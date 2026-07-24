# ANI-2x potential benchmarks
#
# Standalone (prints a table):
#   julia --project=<env> -t 1  benchmark/ani.jl                 # serial baseline
#   julia --project=<env> -t 8  benchmark/ani.jl                 # 8 threads
#
# Thread sweep (records wall-clock vs thread count):
#   for n in 1 2 4 8 12; do JULIA_NUM_THREADS=$n julia --project=<env> benchmark/ani.jl; done
#
# Configuration via environment variables:
#   ANI_SIZES        comma list of atom counts, e.g. "1000,5000,15954"  (default "1000")
#   ANI_FORCES       "true"/"false" — also benchmark forces           (default "true")
#   ANI_ENSEMBLE     "0" for single member, "full" for 8-member        (default "0")
#   ANI_SAMESPECIES  "true"/"false" — run the all-one-species diagnostic (default "true")
#
# BenchmarkTools SUITE (for CI tracking) is still exported as `SUITE`.

using BenchmarkTools, Molly, Lux, HDF5, KernelAbstractions
using StaticArrays, Unitful

# Forces use the analytic path (AtomsCalculators.forces! → compute_ani_forces_ka), available
# whenever Lux + HDF5 are loaded (KernelAbstractions is a core Molly dependency).

const H5_PATH  = joinpath(@__DIR__, "..", "data", "ani_reference", "ani2x.h5")
const PDB_PATH = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")

# ── config ──────────────────────────────────────────────────────────────────
_envbool(k, d) = lowercase(get(ENV, k, d)) in ("1", "true", "yes")
const SIZES       = parse.(Int, split(get(ENV, "ANI_SIZES", "1000"), ","))
const DO_FORCES   = _envbool("ANI_FORCES", "true")
const ENSEMBLE    = get(ENV, "ANI_ENSEMBLE", "0")
const DO_SAMESP   = _envbool("ANI_SAMESPECIES", "true")
_ens_kw() = ENSEMBLE == "full" ? (;) : (; ensemble_idx = parse(Int, ENSEMBLE))

const SUITE = BenchmarkGroup()
SUITE["ANI"] = BenchmarkGroup()

# ── 6mrr loader (parametrized; reuses the PDB column layout) ──────────────────
"""
    load_6mrr(pot, n_max; element=nothing)

Parse up to `n_max` ANI-supported atoms from `6mrr_equil.pdb` and build a
`System` with a `DistanceNeighborFinder`. If `element` is given, every atom is
forced to that element (same-species diagnostic).
"""
function load_6mrr(pot, n_max::Int; element=nothing)
    valid = Set(keys(pot.species_map))
    coords_list = SVector{3,Float64}[]
    elem_list   = String[]
    open(PDB_PATH) do f
        for line in eachline(f)
            (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
            length(line) < 78 && continue
            elem = strip(line[77:78])
            elem in valid || continue
            x = parse(Float64, line[31:38])
            y = parse(Float64, line[39:46])
            z = parse(Float64, line[47:54])
            push!(coords_list, SVector(x, y, z))
            push!(elem_list, isnothing(element) ? elem : element)
            length(elem_list) == n_max && break
        end
    end
    n = length(elem_list)
    nf = DistanceNeighborFinder(
        eligible    = trues(n, n),
        dist_cutoff = (Float64(pot.cutoff) + 1.0) * u"Å",
    )
    sys = System(
        atoms          = [Atom(mass=1.0u"u") for _ in 1:n],
        coords         = [c * u"Å" for c in coords_list],
        boundary       = CubicBoundary(200.0u"Å"),
        atoms_data     = [AtomData(element=e) for e in elem_list],
        general_inters = (ani=pot,),
        neighbor_finder = nf,
        force_units    = u"eV/Å",
        energy_units   = u"eV",
    )
    return sys, n
end

# warm + time helper: returns (median_time_s, allocs_bytes) over a capped run
function timed(f, sys; seconds=20.0, samples=50)
    f(sys)  # warmup (JIT + buffers + neighbor list)
    b = @benchmark $f($sys) samples=samples seconds=seconds evals=1
    return median(b).time / 1e9, median(b).memory
end

function run_bench()
    if !isfile(H5_PATH)
        @warn "ani2x.h5 not found — run test/torchani_reference.py first"
        return
    end
    println("="^72)
    println("ANI-2x benchmark | threads=$(Threads.nthreads()) | ensemble=$(ENSEMBLE)")
    println("sizes=$(SIZES) forces=$(DO_FORCES) samespecies=$(DO_SAMESP)")
    println("="^72)

    pot = ANIPotential(H5_PATH; _ens_kw()...)
    rows = String[]
    push!(rows, rpad("system", 22) * rpad("energy (ms)", 16) * rpad("e-allocs", 14) *
                rpad("forces (ms)", 16) * "f-allocs")

    isfile(PDB_PATH) || (@warn "6mrr_equil.pdb not found"; return)
    for n in SIZES
        sys, nat = load_6mrr(pot, n)
        et, em = timed(potential_energy, sys; seconds = nat > 8000 ? 60.0 : 20.0,
                                              samples = nat > 8000 ? 5 : 30)
        ft, fm = (NaN, 0)
        if DO_FORCES
            ft, fm = timed(forces, sys; seconds = nat > 8000 ? 120.0 : 30.0,
                                        samples = nat > 8000 ? 3 : 15)
        end
        push!(rows, rpad("6mrr_$(nat)", 22) *
                    rpad(string(round(et*1e3, digits=3)), 16) *
                    rpad(string(em), 14) *
                    rpad(DO_FORCES ? string(round(ft*1e3, digits=3)) : "—", 16) *
                    (DO_FORCES ? string(fm) : "—"))
        SUITE["ANI"]["6mrr_$(nat)_PE"] = @benchmarkable potential_energy($sys)
    end

    # same-species diagnostic at the first size (Joe's check: is species branching free?)
    if DO_SAMESP && !isempty(SIZES)
        n = first(SIZES)
        sys_mix, _ = load_6mrr(pot, n)
        sys_one, _ = load_6mrr(pot, n; element="C")
        tm, _ = timed(potential_energy, sys_mix; samples=30)
        to, _ = timed(potential_energy, sys_one; samples=30)
        push!(rows, "")
        push!(rows, "same-species diagnostic @ $n atoms:")
        push!(rows, "  mixed:      $(round(tm*1e3, digits=3)) ms")
        push!(rows, "  all-carbon: $(round(to*1e3, digits=3)) ms")
        push!(rows, "  ratio (mixed/one): $(round(tm/to, digits=3))  " *
                    "(≈1 ⇒ branching is free, no need to sort by species)")
    end

    println()
    foreach(println, rows)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bench()
end
