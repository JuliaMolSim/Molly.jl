# ANI-2x potential benchmarks
# Run with:
#   julia --project benchmark/ani.jl
# or via BenchmarkTools suite:
#   julia --project -e 'using BenchmarkTools; include("benchmark/ani.jl"); run(SUITE["ANI"])'

using BenchmarkTools, Molly, Lux, HDF5
using StaticArrays, Unitful

const SUITE = BenchmarkGroup()
SUITE["ANI"] = BenchmarkGroup()

const H5_PATH = joinpath(@__DIR__, "..", "data", "ani_reference", "ani2x.h5")
const PDB_PATH = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")

if !isfile(H5_PATH)
    @warn "ani2x.h5 not found — run scripts/torchani_reference.py first"
else

# ── N₂ dimer (2 atoms) ─────────────────────────────────────────────────────
pot_n2 = ANIPotential(H5_PATH; ensemble_idx=0)
sys_n2 = let
    atoms      = [Atom(mass=14.0u"u"), Atom(mass=14.0u"u")]
    coords     = [SVector(0.0, 0.0, 0.0)u"Å", SVector(1.1, 0.0, 0.0)u"Å"]
    atoms_data = [AtomData(element="N"), AtomData(element="N")]
    boundary   = CubicBoundary(100.0u"Å")
    System(atoms=atoms, coords=coords, boundary=boundary, atoms_data=atoms_data,
           general_inters=(ani=pot_n2,), force_units=u"eV/Å", energy_units=u"eV")
end

# Warmup to trigger JIT and buffer allocation
potential_energy(sys_n2); forces(sys_n2)

SUITE["ANI"]["N2_potential_energy"] = @benchmarkable potential_energy($sys_n2)
SUITE["ANI"]["N2_forces"]           = @benchmarkable forces($sys_n2)

# ── 6mrr first 50 atoms ─────────────────────────────────────────────────────
if isfile(PDB_PATH)
    pot_prot = ANIPotential(H5_PATH; ensemble_idx=0)
    valid_elements = Set(keys(pot_prot.species_map))

    coords_list = SVector{3,Float64}[]
    elem_list   = String[]
    open(PDB_PATH) do f
        for line in eachline(f)
            (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
            length(line) < 78 && continue
            elem = strip(line[77:78])
            elem in valid_elements || continue
            x = parse(Float64, line[31:38])
            y = parse(Float64, line[39:46])
            z = parse(Float64, line[47:54])
            push!(coords_list, SVector(x, y, z))
            push!(elem_list, elem)
            length(elem_list) == 50 && break
        end
    end
    n = length(elem_list)
    nf_prot = DistanceNeighborFinder(
        eligible    = trues(n, n),
        dist_cutoff = (Float64(pot_prot.cutoff) + 1.0) * u"Å",
    )
    sys_prot = System(
        atoms      = [Atom(mass=1.0u"u") for _ in 1:n],
        coords     = [c * u"Å" for c in coords_list],
        boundary   = CubicBoundary(100.0u"Å"),
        atoms_data = [AtomData(element=e) for e in elem_list],
        general_inters = (ani=pot_prot,),
        neighbor_finder = nf_prot,
        force_units = u"eV/Å",
        energy_units = u"eV",
    )
    potential_energy(sys_prot)  # warmup

    SUITE["ANI"]["6mrr_50atoms_PE"] = @benchmarkable potential_energy($sys_prot)
end

end # isfile check

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running ANI benchmarks...")
    results = run(SUITE["ANI"], verbose=true)
    println("\nSummary:")
    for (name, result) in results
        println("  $name: $(BenchmarkTools.prettytime(median(result).time)) | ",
                "$(BenchmarkTools.prettymemory(median(result).memory))")
    end
end
