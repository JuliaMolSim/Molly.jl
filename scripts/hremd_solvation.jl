##
using Distributed

# One worker process per GPU, each simulating a block of λ states
N_GPUS = 2
addprocs(N_GPUS; exeflags="--project=$(Base.active_project())")
@everywhere using Molly, CUDA
using Unitful
using GLMakie
using Random

##
# --- Simulation Constants ---
CUDA.device!(parse(Int, get(ENV, "MOLLY_CUDA_DEVICE", "0")))
gpu_devices = collect(CUDA.devices())[1:N_GPUS]
FT = Float32
AT = CuArray
Δt = FT(4)u"fs"
T0 = FT(298.15)u"K"
P0 = FT(1)u"bar"
RNG_SEED = 20240520
OUTPUT_PREFIX = "hremd_solvation"

N_LAMBDA_STATES = 20
HREMD_TIME = FT(1.5)u"ns"     # simulated time for each λ state
EXCHANGE_TIME = FT(2)u"ps"
SAMPLE_TIME = FT(4)u"ps"      # one MBAR sample per state, a whole and even number of exchange cycles
EQUIL_FRAC = 0.2              # fraction of the samples of each state discarded before MBAR
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"
CUTOFF = FT(1)u"nm"

cycles_per_sample = Int(round(SAMPLE_TIME / Δt)) / Int(round(EXCHANGE_TIME / Δt))
isinteger(cycles_per_sample) && iseven(Int(cycles_per_sample)) ||
    error("SAMPLE_TIME must be a whole and even number of exchange cycles")

# Experimental hydration free energy of benzene
EXPERIMENT = -3.5u"kJ * mol^-1"

# Replica exchange uses the default random number generator
Random.seed!(RNG_SEED)

# `AbsoluteFESystem` decouples the solute from global λ = 0 (fully coupled) to λ = 1 (fully
# decoupled). The scheduler removes the charges over λ = 0 -> 0.5 with the sterics untouched,
# then the sterics over λ = 0.5 -> 1 using the Beutler LJ soft core. The solute's own
# electrostatics are annihilated with its charges, and intraLJ=true keeps its internal LJ.
# For the GROMACS convention with two PME grids, use GROMACSLambdaABFEScheduler instead.
lambda_schedule = FT.(range(1.0, stop=0.0, length=N_LAMBDA_STATES))
scheduler = DefaultLambdaScheduler(dual=true, intraLJ=true)

# --- Force Field Setup ---
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")
ff = MolecularForceField(
    joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...;
    units=true,
    float_type=FT,
)

##
function build_thermo_states(pdb_file, solute_indices; is_vacuum=false, rng=Random.default_rng())
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : CUTOFF
    nonbonded_method = is_vacuum ? DistanceCutoff(dist_cutoff) : SetupPME()
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing

    sys_base = System(
        pdb_file,
        ff;
        array_type=AT,
        float_type=FT,
        boundary=boundary,
        dist_cutoff=dist_cutoff,
        dist_buffer=FT(0) * u"nm",
        neighbor_finder_type=neighbor_finder_type,
        nonbonded_method=nonbonded_method,
        constraints=:hbonds,
        constraint_algorithm=SetupLINCS(),
        rigid_water=true,
        hydrogen_mass=3
    )

    if is_vacuum
        integrator = Langevin(; dt = Δt,   temperature = T0, friction = FT(1)u"ps^-1", coupling = nothing, remove_CM_motion = 100)
        int_eq     = Langevin(; dt = Δt/2, temperature = T0, friction = FT(1)u"ps^-1", coupling = nothing, remove_CM_motion = 100)
    else
        barostat   = CRescaleBarostat(P0, FT(4)u"ps"; n_steps=200)
        integrator = Langevin(dt = Δt,   temperature = T0, friction = FT(1)u"ps^-1", coupling = (barostat,), remove_CM_motion = 100)
        int_eq     = Langevin(dt = Δt/2, temperature = T0, friction = FT(1)u"ps^-1", coupling = (barostat,), remove_CM_motion = 100)
    end

    minim = SteepestDescentMinimizer(step_size=FT(0.01)u"nm", max_steps=1000)
    simulate!(sys_base, minim)
    random_velocities!(sys_base, T0; rng=rng)

    equil_time = is_vacuum ? VACUUM_EQUIL_TIME : SOLVENT_EQUIL_TIME
    equil_steps = Int(floor(equil_time / Δt))
    simulate!(sys_base, int_eq, 10_000; rng=rng)
    simulate!(sys_base, integrator, equil_steps; rng=rng)

    # The barostat changes the box during equilibration, so check that the minimum image
    # convention still holds for the cutoff
    if !is_vacuum
        edge = minimum(Molly.box_sides(sys_base.boundary))
        edge >= 2 * dist_cutoff || error("equilibrated box edge $edge is smaller than twice the cutoff")
    end

    thermo_states = ThermoState[]

    for λ in lambda_schedule
        sys_w = AbsoluteFESystem(
            sys_base,
            FT(λ),
            solute_indices;
            scheduler = scheduler,
            LJsoftcore = "beutler",
            Csoftcore = "scaled",
            array_type = AT,
            float_type = FT,
        )

        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))
    end

    return thermo_states, sys_base
end

# All replicas start from the equilibrated configuration. `simulate_remd!` returns a new
# `ReplicaSystem`, and after each call the configuration currently in each state is stored.
function run_hremd_leg(thermo_states, sys_base)
    K = length(thermo_states)
    repsys = ReplicaSystem(
        thermo_states,
        [copy(sys_base.coords) for _ in 1:K];
        replica_velocities = [copy(sys_base.velocities) for _ in 1:K],
        replica_boundaries = [sys_base.boundary for _ in 1:K],
        reuse_neighbors = true,
    )
    sim = ReplicaExchangeMD(dt = Δt, exchange_time = EXCHANGE_TIME)

    chunk_steps = Int(round(SAMPLE_TIME / Δt))
    n_chunks = Int(floor(HREMD_TIME / Δt)) ÷ chunk_steps
    coords_k = [Any[] for _ in 1:K]
    boundaries_k = [Any[] for _ in 1:K]

    for _ in 1:n_chunks
        repsys = simulate_remd!(repsys, sim, chunk_steps;
                                gpu_devices = gpu_devices, show_progress = false)

        for k in 1:K
            r = repsys.state_indices[k]
            push!(coords_k[k], Array(repsys.replica_coords[r]))
            push!(boundaries_k[k], repsys.replica_boundaries[r])
        end
    end

    return coords_k, boundaries_k, repsys.exchange_logger
end

function hremd_free_energies(thermo_states, coords_k, boundaries_k)
    K = length(thermo_states)
    coords_dev = [[Molly.to_device(c, AT) for c in v] for v in coords_k]
    mbar_gen = assemble_mbar_inputs(coords_dev, boundaries_k, thermo_states)

    # Index of every sample within its own state, to discard the start of each state
    sample_idx = zeros(Int, length(mbar_gen.win_of))
    seen = zeros(Int, K)
    for n in eachindex(mbar_gen.win_of)
        k = mbar_gen.win_of[n]
        seen[k] += 1
        sample_idx[n] = seen[k]
    end
    kept = [n for n in eachindex(sample_idx)
            if sample_idx[n] > floor(Int, EQUIL_FRAC * mbar_gen.N[mbar_gen.win_of[n]])]

    win_of = mbar_gen.win_of[kept]
    N_counts = [count(==(k), win_of) for k in 1:K]
    F_k, logN = iterate_mbar(mbar_gen.u[kept, :], win_of, N_counts)

    return F_k
end

# Accepted exchanges for each neighbouring pair of states, divided by the attempts on that pair.
# Pairs alternate between exchange cycles, so each pair is attempted every other cycle.
function pair_acceptance(exchange_logger, K)
    n_cycles = Int(floor(HREMD_TIME / Δt)) ÷ Int(round(SAMPLE_TIME / Δt)) *
               (Int(round(SAMPLE_TIME / Δt)) ÷ Int(round(EXCHANGE_TIME / Δt)))
    accepted = zeros(Int, K - 1)
    for (n, m) in exchange_logger.indices
        accepted[min(n, m)] += 1
    end
    return accepted ./ (n_cycles ÷ 2)
end

##

solute_idx = 1:12

# 4 nm cube of TIP3P water with 0.15 M NaCl, large enough for the 1 nm cutoff
thermo_solv, base_solv = build_thermo_states(
    joinpath(data_dir, "benzene_solv_4.pdb"),
    solute_idx;
    is_vacuum=false,
    rng=MersenneTwister(RNG_SEED),
)
println()

##
coords_solv, boundaries_solv, log_solv = run_hremd_leg(thermo_solv, base_solv)

##
thermo_vac, base_vac = build_thermo_states(
    joinpath(data_dir, "benzene_vac.pdb"),
    solute_idx;
    is_vacuum=true,
    rng=MersenneTwister(RNG_SEED + 1),
)
println()

##
coords_vac, boundaries_vac, log_vac = run_hremd_leg(thermo_vac, base_vac)

##
f_solv = hremd_free_energies(thermo_solv, coords_solv, boundaries_solv)
f_vac  = hremd_free_energies(thermo_vac, coords_vac, boundaries_vac)

lambda_plot = reverse(lambda_schedule)
f_solv_plot = reverse(f_solv)
f_vac_plot = reverse(f_vac)

##
fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
          title = L"\textbf{Alchemical Free Energy}",
          xlabel = L"\textbf{\lambda}",
          ylabel = L"\textbf{F / k_{B}T}",
          xlabelsize = 20, ylabelsize = 20,
          titlesize = 24,
          xlabelfont = :bold, ylabelfont = :bold,
          xticklabelsize = 18, yticklabelsize = 18)

lines!(
    ax_fe,
    lambda_plot, f_solv_plot;
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated"
)

lines!(
    ax_fe,
    lambda_plot, f_vac_plot;
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum"
)

axislegend(
    position = :rt,
    labelsize = 24
)

display(fig_fe)

save("$(OUTPUT_PREFIX)_profile.png", fig_fe)

##
# A pair of neighbouring states that never exchanges splits the ladder in two and
# invalidates the MBAR estimate
acc_solv = pair_acceptance(log_solv, N_LAMBDA_STATES)
acc_vac  = pair_acceptance(log_vac, N_LAMBDA_STATES)

fig_acc = Figure(size = (720, 720))

ax_acc = Axis(
    fig_acc[1,1],
    title = L"\textbf{Exchange Acceptance}",
    xlabel = L"\textbf{State Pair}",
    ylabel = L"\textbf{Acceptance}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18
)

scatterlines!(
    ax_acc,
    1:(N_LAMBDA_STATES - 1), acc_solv;
    color = :royalblue,
    linewidth = 3,
    markersize = 12,
    label = "Solvated"
)

scatterlines!(
    ax_acc,
    1:(N_LAMBDA_STATES - 1), acc_vac;
    color = :firebrick,
    linewidth = 3,
    markersize = 12,
    label = "Vacuum"
)

ylims!(ax_acc, 0, 1.05)

axislegend(
    position = :rb,
    labelsize = 24
)

display(fig_acc)

save("$(OUTPUT_PREFIX)_acceptance.png", fig_acc)

##

# The first state is decoupled (λ = 1) and the last is coupled (λ = 0)
dG_solv = f_solv[1] - f_solv[end]
dG_vac  = f_vac[1]  - f_vac[end]

# Hydration free energy from the two-leg cycle
dG = dG_vac - dG_solv

##
println("=========================================")
println("Annihilation in solvent (kBT): ", dG_solv)
println("Annihilation in vacuum (kBT):  ", dG_vac)
println("Hydration Free Energy (kBT):   ", dG)
println("=========================================")

##

beta = thermo_solv[1].beta

println("=========================================")
println("Annihilation in solvent (kJ mol^-1): ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):  ", dG_vac / beta)
println("Hydration Free Energy (kJ mol^-1):   ", dG / beta)
println("Experiment (kJ mol^-1):              ", EXPERIMENT)
println("Lowest pair acceptance in solvent:   ", minimum(acc_solv))
println("=========================================")
