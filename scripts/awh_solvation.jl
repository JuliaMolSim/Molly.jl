##
using Molly
using CUDA
using Unitful
using GLMakie
using Random

##
# --- Simulation Constants ---
CUDA.device!(parse(Int, get(ENV, "MOLLY_CUDA_DEVICE", "0")))
FT = Float32
AT = CuArray
Δt = FT(4)u"fs"
T0 = FT(298.15)u"K"
P0 = FT(1)u"bar"
RNG_SEED = 20240520
OUTPUT_PREFIX = "awh_solvation"

N_LAMBDA_STATES = 20
N_MD_STEPS = 50
AWH_TIME = FT(30)u"ns"
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"
CUTOFF = FT(1)u"nm"

# Experimental hydration free energy of benzene, -3.67 ± 0.05 kJ/mol (Kashefolgheta et al. 2020,
# https://pubs.acs.org/jctcce/article/16/12/7556/617259/Evaluating-Classical-Force-Fields-against)
EXPERIMENT = -3.67u"kJ * mol^-1"

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
function awh_solvation_loggers(is_vacuum::Bool)
    if is_vacuum
        return ()
    else
        return (
            vol = VolumeLogger(1000),
            trj = TrajectoryWriter(1000, "$(OUTPUT_PREFIX)_solvated.dcd"),
        )
    end
end

function save_state_histogram(state_sets, labels, bins, path)
    fig = Figure(size = (720, 720))
    ax = Axis(
        fig[1, 1],
        title = L"\textbf{Visited States}",
        xlabel = L"\textbf{State Index}",
        ylabel = L"\textbf{PDF}",
        xlabelsize = 20,
        ylabelsize = 20,
        titlesize = 24,
        xlabelfont = :bold,
        ylabelfont = :bold,
        xticklabelsize = 18,
        yticklabelsize = 18,
    )

    colors = (:royalblue, :firebrick, :seagreen, :darkorange)
    for (i, states) in pairs(state_sets)
        isempty(states) && continue
        hist!(
            ax,
            states;
            bins = bins,
            color = colors[mod1(i, length(colors))],
            alpha = 0.35,
            strokewidth = 1,
            strokecolor = :black,
            normalization = :pdf,
            label = labels[i],
        )
    end

    axislegend(position = :rt, labelsize = 20)
    display(fig)
    save(path, fig)
end

awh_visited_states(state) = collect(state.stats.active_λ)

function setup_alchemical_awh(pdb_file, solute_indices; is_vacuum=false, rng=Random.default_rng())
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : CUTOFF
    nonbonded_method = is_vacuum ? DistanceCutoff(dist_cutoff) : SetupPME()
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing
    awh_loggers = awh_solvation_loggers(is_vacuum)

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

    awh_state = AWHState(thermo_states; reuse_neighbors=true)
    awh_sim = AWHSimulation(
        awh_state;
        num_md_steps = N_MD_STEPS,
        update_freq = 1,
        well_tempered_factor = FT(Inf),
        log_freq = 10,
        loggers = awh_loggers,
    )

    return awh_state, awh_sim
end

##

solute_idx = 1:12

# 4 nm cube of TIP3P water with 0.15 M NaCl, large enough for the 1 nm cutoff
awh_state_solv, awh_sim_solv = setup_alchemical_awh(
    joinpath(data_dir, "benzene_solv.pdb"),
    solute_idx;
    is_vacuum=false,
    rng=MersenneTwister(RNG_SEED),
)
println()

##
awh_state_vac, awh_sim_vac = setup_alchemical_awh(
    joinpath(data_dir, "benzene_vac.pdb"),
    solute_idx;
    is_vacuum=true,
    rng=MersenneTwister(RNG_SEED + 1),
)
println()

##
awh_steps = Int(floor(AWH_TIME / Δt))

Random.seed!(RNG_SEED + 2)
simulate!(awh_sim_solv, awh_steps)

##
Random.seed!(RNG_SEED + 3)
simulate!(awh_sim_vac, awh_steps)

##
f_solv = awh_state_solv.f
f_vac  = awh_state_vac.f

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

deltaF_solv = awh_state_solv.stats.max_delta_f_history
deltaF_vac  = awh_state_vac.stats.max_delta_f_history

iter_solv = awh_state_solv.stats.step_indices
iter_vac = awh_state_vac.stats.step_indices

##

fig_df = Figure(size = (720, 720))

ax_df = Axis(
    fig_df[1,1],
    title = L"\textbf{Max. $\Delta$F}",
    xlabel = L"\textbf{Iteration}",
    ylabel = L"\textbf{log_{10}($\Delta$F)}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18
)

lines!(
    ax_df,
    iter_solv, log10.(deltaF_solv),
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated"
)

lines!(
    ax_df,
    iter_vac, log10.(deltaF_vac),
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

display(fig_df)

save("$(OUTPUT_PREFIX)_convergence.png", fig_df)

##
state_bins = 0.5:1:(N_LAMBDA_STATES + 0.5)
save_state_histogram(
    [awh_visited_states(awh_state_solv), awh_visited_states(awh_state_vac)],
    ["Solvated", "Vacuum"],
    state_bins,
    "$(OUTPUT_PREFIX)_states.png",
)

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

beta = awh_state_solv.state_space.betas[1]

println("=========================================")
println("Annihilation in solvent (kJ mol^-1): ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):  ", dG_vac / beta)
println("Hydration Free Energy (kJ mol^-1):   ", dG / beta)
println("Experiment (kJ mol^-1):              ", EXPERIMENT)
println("=========================================")

#=
Result of a run with these settings:
=========================================
Annihilation in solvent (kJ mol^-1): -6.477
Annihilation in vacuum (kJ mol^-1):  -9.984
Hydration Free Energy (kJ mol^-1):   -3.507
Experiment (kJ mol^-1):              -3.5 kJ mol^-1
=========================================
=#
