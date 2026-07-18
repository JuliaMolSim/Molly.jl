# Full-system 6mrr NVE molecular dynamics with ANI forces computed ON THE GPU (Apple Metal),
# writing a DCD trajectory. Unlike ani_trajectory.jl (CPU Enzyme forces, small slice), this
# drives Molly's simulate! with a thin wrapper potential whose forces!/potential_energy call
# the on-device path (compute_ani_forces_ka / compute_ani_energy_ka). Molly still owns the
# integration, neighbour rebuilds, and DCD writing — only the force/energy evaluation is on GPU,
# which is what makes the full 15,954-atom system tractable (~0.4 s/force vs tens of s on CPU).
#
#   julia --project=<env> -t8 benchmark/ani_trajectory_metal.jl
# Config: ANI_TRAJ_N (default 15954 = full 6mrr), ANI_TRAJ_STEPS (1000), ANI_TRAJ_DT_FS (0.5),
#         ANI_TRAJ_LOG (20), ANI_TRAJ_TEMP (300), ANI_TRAJ_NBR (neighbour rebuild interval, 10).

using Molly, Lux, HDF5, KernelAbstractions, Metal, StaticArrays, Unitful, Random, LinearAlgebra
const AtomsCalculators = Molly.AtomsCalculators   # Molly's AD-calculator interface (transitive dep)

const ELEM_MASS = Dict("H"=>1.008u"u", "C"=>12.011u"u", "N"=>14.007u"u", "O"=>15.999u"u",
                       "S"=>32.06u"u", "F"=>18.998u"u", "Cl"=>35.45u"u")
const H5_PATH  = joinpath(@__DIR__, "..", "data", "ani_reference", "ani2x.h5")
const PDB_PATH = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")

# Wrapper potential: routes force/energy evaluation to the Metal on-device kernels. Holds the
# device species vector and a Float32 boundary so nothing per-call needs the CPU ANIPotential.
struct MetalANI{P, S, B}
    pot   :: P
    n_sp  :: Int
    sM    :: S    # device species indices (MtlArray{Int32})
    bdy32 :: B    # Float32 boundary for the kernels
end

# Upload current coords (Å, unit-stripped Float32) to the device once per call.
@inline dev_coords(sys) = MtlArray([SVector{3,Float32}(ustrip.(u"Å", c)) for c in sys.coords])
@inline nbrs_of(sys, kwargs) = (n = get(kwargs, :neighbors, nothing);
                                n === nothing ? Molly.find_neighbors(sys) : n)

function AtomsCalculators.forces!(fs, sys, inter::MetalANI; kwargs...)
    nbrs = nbrs_of(sys, kwargs)
    F = Molly.compute_ani_forces_ka(dev_coords(sys), inter.sM, inter.pot, inter.n_sp;
            backend = MetalBackend(), neighbors = nbrs, boundary = inter.bdy32)   # eV/Å
    @inbounds for i in eachindex(fs)
        fs[i] += SVector{3,Float64}(F[i]) * u"eV/Å"
    end
    return fs
end

function AtomsCalculators.potential_energy(sys, inter::MetalANI; kwargs...)
    nbrs = nbrs_of(sys, kwargs)
    E = Molly.compute_ani_energy_ka(dev_coords(sys), inter.sM, inter.pot, inter.n_sp;
            backend = MetalBackend(), neighbors = nbrs, boundary = inter.bdy32)    # eV
    return E * u"eV"
end

n_max  = parse(Int,     get(ENV, "ANI_TRAJ_N",     "15954"))
nsteps = parse(Int,     get(ENV, "ANI_TRAJ_STEPS", "1000"))
dt_fs  = parse(Float64, get(ENV, "ANI_TRAJ_DT_FS", "0.5"))
logint = parse(Int,     get(ENV, "ANI_TRAJ_LOG",   "20"))
temp0  = parse(Float64, get(ENV, "ANI_TRAJ_TEMP",  "300"))
nbrint = parse(Int,     get(ENV, "ANI_TRAJ_NBR",   "10"))

pot   = ANIPotential(H5_PATH; ensemble_idx=0)      # single member
n_sp  = length(pot.species_map)
valid = Set(keys(pot.species_map))
println("Metal functional: ", Metal.functional())

coords = SVector{3,Float64}[]; elems = String[]
open(PDB_PATH) do f
    for line in eachline(f)
        (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
        length(line) < 78 && continue
        e = strip(line[77:78]); e in valid || continue
        push!(coords, SVector(parse(Float64, line[31:38]), parse(Float64, line[39:46]),
                              parse(Float64, line[47:54])))
        push!(elems, e)
        length(elems) == n_max && break
    end
end
n = length(coords)

sM     = MtlArray(Int32.([pot.species_map[e] for e in elems]))
metal  = MetalANI(pot, n_sp, sM, CubicBoundary(200.0f0))
atoms  = [Atom(mass = ELEM_MASS[e]) for e in elems]
nf     = DistanceNeighborFinder(eligible = trues(n, n),
                                dist_cutoff = (Float64(pot.cutoff) + 1.0) * u"Å",
                                n_steps = nbrint)

traj = joinpath(@__DIR__, "traj_6mrr_$(n)atoms_metal.dcd")
isfile(traj) && rm(traj)

sys = System(
    atoms           = atoms,
    coords          = [c * u"Å" for c in coords],
    boundary        = CubicBoundary(200.0u"Å"),
    atoms_data      = [AtomData(element = e) for e in elems],
    general_inters  = (ani = metal,),
    neighbor_finder = nf,
    loggers = (
        energy = TotalEnergyLogger(typeof(1.0u"eV"), logint),
        writer = TrajectoryWriter(logint, traj),
    ),
    force_units  = u"eV/Å",
    energy_units = u"eV",
)

random_velocities!(sys, temp0 * u"K"; rng = MersenneTwister(42))
E0 = total_energy(sys)
println("="^60)
println("6mrr ANI NVE on Metal: $n atoms, $nsteps steps, dt=$dt_fs fs, T0=$temp0 K")
println("neighbour rebuild every $nbrint steps | log/write every $logint → $(nsteps ÷ logint) frames")
println("E0 = ", E0)
println("="^60)

simulate!(deepcopy(sys), VelocityVerlet(dt = dt_fs * u"fs"), 2; run_loggers=false)  # warmup/compile
t = @elapsed simulate!(sys, VelocityVerlet(dt = dt_fs * u"fs"), nsteps)

Ef    = total_energy(sys)
drift = abs(ustrip(Ef - E0)) / abs(ustrip(E0))
println("finished in $(round(t, digits=1)) s  ($(round(1000t/nsteps, digits=1)) ms/step)")
println("E_final = ", Ef)
println("relative energy drift |ΔE|/|E0| = ", drift)
println("trajectory written: $traj  ($(round(filesize(traj)/1024/1024, digits=2)) MiB)")
