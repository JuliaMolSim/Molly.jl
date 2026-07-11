# 6mrr NVE molecular dynamics with the ANI potential, writing a DCD trajectory.
#
# Run:
#   julia --project=<env> -t 8 benchmark/ani_trajectory.jl
#
# Config (env vars):
#   ANI_TRAJ_N       atoms from 6mrr_equil.pdb            (default 300)
#   ANI_TRAJ_STEPS   number of MD steps                   (default 2000)
#   ANI_TRAJ_DT_FS   timestep in fs                       (default 0.5)
#   ANI_TRAJ_LOG     log/write interval in steps          (default 20)
#   ANI_TRAJ_TEMP    initial temperature in K             (default 300)
#
# Note on cost: ANI forces run through single-pass Enzyme reverse-mode AD (seconds/step at
# ~1000+ atoms on CPU), so a full-6mrr multi-thousand-step run is hours. This script targets a
# tractable slice to demonstrate a stable end-to-end ANI NVE trajectory; on-device GPU forces
# (future) would make full-system ANI MD practical.

using Molly, Lux, HDF5, StaticArrays, Unitful, Random, LinearAlgebra
try
    @eval using Enzyme   # single-pass AD forces (else forces! falls back to finite differences)
catch
    @warn "Enzyme not loaded — forces use the slow finite-difference fallback"
end

# Standard atomic masses for the 7 ANI-2x elements (u). Real masses matter for dynamics —
# the energy/force benchmarks use a placeholder 1 u, which is fine for timing but not MD.
const ELEM_MASS = Dict("H"=>1.008u"u", "C"=>12.011u"u", "N"=>14.007u"u", "O"=>15.999u"u",
                       "S"=>32.06u"u", "F"=>18.998u"u", "Cl"=>35.45u"u")

const H5_PATH  = joinpath(@__DIR__, "..", "data", "ani_reference", "ani2x.h5")
const PDB_PATH = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")

n_max  = parse(Int,     get(ENV, "ANI_TRAJ_N",     "300"))
nsteps = parse(Int,     get(ENV, "ANI_TRAJ_STEPS", "2000"))
dt_fs  = parse(Float64, get(ENV, "ANI_TRAJ_DT_FS", "0.5"))
logint = parse(Int,     get(ENV, "ANI_TRAJ_LOG",   "20"))
temp0  = parse(Float64, get(ENV, "ANI_TRAJ_TEMP",  "300"))

pot   = ANIPotential(H5_PATH; ensemble_idx=0)      # single member for speed
valid = Set(keys(pot.species_map))

coords = SVector{3,Float64}[]; elems = String[]
open(PDB_PATH) do f
    for line in eachline(f)
        (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
        length(line) < 78 && continue
        e = strip(line[77:78]); e in valid || continue
        push!(coords, SVector(parse(Float64, line[31:38]),
                              parse(Float64, line[39:46]),
                              parse(Float64, line[47:54])))
        push!(elems, e)
        length(elems) == n_max && break
    end
end
n = length(coords)

# Real element-derived masses (the benchmarks use placeholder 1 u — unsuitable for dynamics).
atoms = [Atom(mass = ELEM_MASS[e]) for e in elems]
nf    = DistanceNeighborFinder(eligible = trues(n, n),
                               dist_cutoff = (Float64(pot.cutoff) + 1.0) * u"Å")

traj = joinpath(@__DIR__, "traj_6mrr_$(n)atoms.dcd")
isfile(traj) && rm(traj)   # TrajectoryWriter appends

sys = System(
    atoms          = atoms,
    coords         = [c * u"Å" for c in coords],
    boundary       = CubicBoundary(200.0u"Å"),
    atoms_data     = [AtomData(element = e) for e in elems],
    general_inters = (ani = pot,),
    neighbor_finder = nf,
    loggers = (
        energy = TotalEnergyLogger(typeof(1.0u"eV"), logint),   # match the system's eV units
        writer = TrajectoryWriter(logint, traj),
    ),
    force_units  = u"eV/Å",
    energy_units = u"eV",
)

random_velocities!(sys, temp0 * u"K"; rng = MersenneTwister(42))
E0 = total_energy(sys)

println("="^60)
println("6mrr ANI NVE: $n atoms, $nsteps steps, dt=$dt_fs fs, T0=$temp0 K")
println("log/write every $logint steps → $(nsteps ÷ logint) frames")
println("E0 = ", E0)
println("="^60)

# Warmup: compile forces + build neighbours, without logging.
simulate!(deepcopy(sys), VelocityVerlet(dt = dt_fs * u"fs"), 2; run_loggers=false)

t = @elapsed simulate!(sys, VelocityVerlet(dt = dt_fs * u"fs"), nsteps)

Ef    = total_energy(sys)
drift = abs(ustrip(Ef - E0)) / abs(ustrip(E0))
es    = values(sys.loggers.energy)   # energy trace over the run
println("finished in $(round(t, digits=1)) s  ($(round(1000t/nsteps, digits=1)) ms/step)")
println("E_final = ", Ef)
println("relative energy drift |ΔE|/|E0| = ", drift)
if !isempty(es)
    esu = ustrip.(es)
    println("energy trace: $(length(es)) frames, min=$(round(minimum(esu),digits=3)) ",
            "max=$(round(maximum(esu),digits=3)) eV")
end
println("trajectory written: $traj  ($(round(filesize(traj)/1024, digits=1)) KiB)")
