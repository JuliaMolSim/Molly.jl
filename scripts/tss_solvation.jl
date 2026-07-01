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
T0 = FT(310)u"K"
P0 = FT(1)u"bar"
RNG_SEED = 20240520
OUTPUT_PREFIX = "tss_solvation"

N_LAMBDA_STATES = 20
TSS_WINDOW_SIZE = 4
N_MD_STEPS = 50
SELF_ADJUSTMENT_STEPS = 5
N_REPLICAS = 4
TSS_TIME = FT(15)u"ns"
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"

# Annihilate by running from full interactions at λ=1 to decoupled at λ=0.
# InsertRole with DefaultLambdaScheduler keeps sterics fully on while
# electrostatics are removed from λ=1 -> 0.5, then removes sterics from
# λ=0.5 -> 0 using OpenFE-style charge scaling. Sterics use LJ soft core.
lambda_schedule = FT.(range(1.0, stop=0.0, length=N_LAMBDA_STATES))

# --- Force Field Setup ---
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")
ff = MolecularForceField(FT, joinpath.(ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)

##
function tss_solvation_loggers(is_vacuum::Bool, replica_i=nothing)
    suffix = isnothing(replica_i) ? "" : "_replica_$(replica_i)"
    traj_path = "$(OUTPUT_PREFIX)_solvated$(suffix).dcd"

    if is_vacuum
        return ()
    else
        return (
            trj = TrajectoryWriter(1000, traj_path),
        )
    end
end

tss_solvation_replica_rngs(seed::Integer) =
    [MersenneTwister(seed + replica_i - 1) for replica_i in 1:N_REPLICAS]

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

function tss_visited_states(state)
    replica_states = state.stats.replica_visited_states
    return isempty(replica_states) ?
        collect(state.stats.visited_state) :
        collect(Iterators.flatten(replica_states))
end

function alchemical_coulomb_scaled(coul::CoulombEwald)
    return CoulombEwaldScaled(
        dist_cutoff = coul.dist_cutoff,
        error_tol = coul.error_tol,
        use_neighbors = coul.use_neighbors,
        scheduler = Molly.DefaultLambdaScheduler(),
        weight_special = coul.weight_special,
        coulomb_const = coul.coulomb_const,
        approximate_erfc = coul.approximate_erfc,
    )
end

function alchemical_coulomb_scaled(coul::Coulomb)
    return CoulombScaled(
        cutoff = coul.cutoff,
        use_neighbors = coul.use_neighbors,
        scheduler = Molly.DefaultLambdaScheduler(),
        weight_special = coul.weight_special,
        coulomb_const = coul.coulomb_const,
    )
end

function alchemical_ewald_exclusion_data(data::Molly.EwaldExclusionData, scheduler)
    return Molly.EwaldExclusionData(
        data.dist_cutoff;
        error_tol = data.error_tol,
        ϵr = data.ϵr,
        scheduler = scheduler,
    )
end

function alchemical_lj_softcore(lj::LennardJones)
    return LennardJonesSoftCoreGapsys(
        cutoff = lj.cutoff,
        α = FT(0.85),
        use_neighbors = lj.use_neighbors,
        shortcut = lj.shortcut,
        σ_mixing = lj.σ_mixing,
        ϵ_mixing = lj.ϵ_mixing,
        scheduler = Molly.DefaultLambdaScheduler(),
        weight_special = lj.weight_special,
    )
end

function rebuild_alchemical_general_inters(sys_base, atoms_dev, lj_sc, coul_scaled)
    rebuilt = Any[]

    for inter in sys_base.general_inters
        if inter isa PME
            # Rebuild PME using the λ/role-modified atoms.
            # fixed_charges=false avoids reusing charge sums cached from the λ=1 system.
            push!(rebuilt, PME(
                inter.dist_cutoff,
                atoms_dev,
                sys_base.boundary;
                error_tol = inter.error_tol,
                order = inter.order,
                ϵr = inter.ϵr,
                fixed_charges = false,
                scheduler = coul_scaled.scheduler,
                grad_safe = inter.grad_safe,
            ))

        elseif inter isa LJDispersionCorrection

            push!(rebuilt, LJDispersionCorrection(
                atoms_dev,
                lj_sc.cutoff.dist_cutoff,
                lj_sc.σ_mixing,
                lj_sc.ϵ_mixing,
                lj_sc.λ_mixing,
                lj_sc.scheduler,
            ))

        else
            # Keep unrelated general interactions, but avoid sharing mutable state.
            push!(rebuilt, deepcopy(inter))
        end
    end

    return tuple(rebuilt...)
end

function rebuild_alchemical_specific_inter_lists(sys_base, coul_scaled)
    rebuilt = Any[]

    for inter_list in sys_base.specific_inter_lists
        if inter_list isa InteractionList2Atoms && inter_list.data isa Molly.EwaldExclusionData
            data = alchemical_ewald_exclusion_data(inter_list.data, coul_scaled.scheduler)
            push!(rebuilt, InteractionList2Atoms(
                inter_list.is,
                inter_list.js,
                deepcopy(inter_list.inters),
                copy(inter_list.types),
                data,
            ))
        else
            push!(rebuilt, deepcopy(inter_list))
        end
    end

    return tuple(rebuilt...)
end

function setup_alchemical_tss(pdb_file, solute_indices; is_vacuum=false, rng=Random.default_rng())
    nonbonded_method = is_vacuum ? :none : :pme
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : FT(1) * u"nm"
    dist_buffer = is_vacuum ? FT(0) * u"nm" : FT(0.2) * u"nm"
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing
    replica_loggers = [
        tss_solvation_loggers(is_vacuum, N_REPLICAS == 1 ? nothing : replica_i)
        for replica_i in 1:N_REPLICAS
    ]

    sys_base = System(
        pdb_file,
        ff;
        array_type=AT,
        boundary=boundary,
        dist_cutoff=dist_cutoff,
        dist_buffer=dist_buffer,
        neighbor_finder_type=neighbor_finder_type,
        nonbonded_method=nonbonded_method,
        constraints=:hbonds,
        constraint_algorithm=LINCS,
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

    p_inters = sys_base.pairwise_inters
    idx_lj   = findfirst(x -> x isa LennardJones, p_inters)
    idx_coul = findfirst(x -> x isa Union{Coulomb, CoulombEwald}, p_inters)
    isnothing(idx_lj) && error("could not find LennardJones pairwise interaction")
    isnothing(idx_coul) && error("could not find a Coulomb pairwise interaction")

    lj_sc = alchemical_lj_softcore(p_inters[idx_lj])
    cl_scaled = alchemical_coulomb_scaled(p_inters[idx_coul])

    atoms_cpu = Molly.from_device(sys_base.atoms)
    thermo_states = ThermoState[]

    for λ in lambda_schedule
        acopy = Atom[]
        for a in atoms_cpu
            if a.index ∈ solute_indices
                # Only update the global lambda; the scheduler handles the component logic
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(λ), Molly.InsertRole))
            else
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(a.λ), a.alch_role))
            end
        end

        atoms_dev = Molly.to_device([acopy...], AT)

        general_inters = rebuild_alchemical_general_inters(
            sys_base,
            atoms_dev,
            lj_sc,
            cl_scaled,
        )
        specific_inter_lists = rebuild_alchemical_specific_inter_lists(sys_base, cl_scaled)

        sys_w = System(
            deepcopy(sys_base);
            atoms = atoms_dev,
            pairwise_inters = (lj_sc, cl_scaled),
            general_inters = general_inters,
            specific_inter_lists = specific_inter_lists,
        )

        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))
    end

    tss_graph = Molly.tss_grid_graph(
        (length(lambda_schedule),);
        window_size = (TSS_WINDOW_SIZE,),
        periodic    = (false,)
    )

    tss_state = TSSState(
        thermo_states;
        graph = tss_graph,
        history_forgetting = TSSHistoryForgetting(alpha = FT(0.2), n_epochs = 16),
        adaptive_gamma = :covdet,
    )

    total_steps = Int(floor(TSS_TIME / Δt))
    n_cycles = Int(floor(total_steps / (SELF_ADJUSTMENT_STEPS * N_MD_STEPS)))
    first_states = N_REPLICAS == 1 ? [1] : round.(Int, range(1, length(lambda_schedule); length=N_REPLICAS))
    logger_kwargs = N_REPLICAS == 1 ?
        (; loggers = only(replica_loggers)) :
        (; replica_loggers = replica_loggers)

    tss_sim = TSSSimulation(
        tss_state;
        n_md_steps = N_MD_STEPS,
        n_cycles = n_cycles,
        self_adjustment_steps = SELF_ADJUSTMENT_STEPS,
        n_replicas = N_REPLICAS,
        first_states = first_states,
        logger_kwargs...,
        log_freq = 10,
    )

    return tss_state, tss_sim
end

##

solute_idx = 1:9

tss_state_solv, tss_sim_solv = setup_alchemical_tss(
    joinpath(data_dir, "ethanol_solv.pdb"),
    solute_idx;
    is_vacuum=false,
    rng=MersenneTwister(RNG_SEED),
)
println()
##
tss_state_vac, tss_sim_vac   = setup_alchemical_tss(
    joinpath(data_dir, "ethanol_vac.pdb"),
    solute_idx;
    is_vacuum=true,
    rng=MersenneTwister(RNG_SEED + 1),
)
println()

##
simulate!(
    tss_sim_solv;
    rng=MersenneTwister(RNG_SEED + 2),
    replica_rngs=tss_solvation_replica_rngs(RNG_SEED + 20),
    replica_parallel=:auto,
)

##
simulate!(
    tss_sim_vac;
    rng=MersenneTwister(RNG_SEED + 3),
    replica_rngs=tss_solvation_replica_rngs(RNG_SEED + 40),
    replica_parallel=:auto,
)


##
jk_solv = tss_free_energy_uncertainties(tss_state_solv)
jk_vac  = tss_free_energy_uncertainties(tss_state_vac)

##
f_solv = jk_solv.free_energies
f_vac  = jk_vac.free_energies

se_solv = jk_solv.standard_errors
se_vac  = jk_vac.standard_errors

lambda_plot = reverse(lambda_schedule)
f_solv_plot = reverse(f_solv)
f_vac_plot = reverse(f_vac)
se_solv_plot = reverse(se_solv)
se_vac_plot = reverse(se_vac)

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
band!(
    ax_fe,
    lambda_plot, f_solv_plot - 3 .* se_solv_plot, f_solv_plot + 3 .* se_solv_plot;
    color = :royalblue,
    alpha = 0.45
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
band!(
    ax_fe,
    lambda_plot, f_vac_plot - 3 .* se_vac_plot, f_vac_plot + 3 .* se_vac_plot;
    color = :firebrick,
    alpha = 0.45
)

axislegend(
    position = :rt,
    labelsize = 24
)

display(fig_fe)

save("$(OUTPUT_PREFIX)_profile.png", fig_fe)


##

deltaF_solv = tss_state_solv.stats.max_abs_delta_f
deltaF_vac  = tss_state_vac.stats.max_abs_delta_f

iter_solv = tss_state_solv.stats.iterations
iter_vac = tss_state_vac.stats.iterations

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

ylims!(ax_df, -5, 0.5)


display(fig_df)

save("$(OUTPUT_PREFIX)_convergence.png", fig_df)

##
state_bins = 0.5:1:(N_LAMBDA_STATES + 0.5)
save_state_histogram(
    [tss_visited_states(tss_state_solv), tss_visited_states(tss_state_vac)],
    ["Solvated", "Vacuum"],
    state_bins,
    "$(OUTPUT_PREFIX)_states.png",
)

##

dG_solv = f_solv[end] - f_solv[1]
dG_vac  = f_vac[end]  - f_vac[1]

# Standard gas state (1 bar) volume per molecule at T0
V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)

# Standard solution state (1 M = 1 mol/L) volume per molecule
V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))

# Analytical standard state correction
dG_std_corr = log(V_gas / V_std)

# Final Solvation Free Energy (2-Leg Cycle)
dG = dG_vac - dG_solv + dG_std_corr
dG_se = hypot(se_solv[end], se_vac[end])

##
println("=========================================")
println("Annihilation in solvent (kBT):   ", dG_solv)
println("Annihilation in vacuum (kBT):    ", dG_vac)
println("Standard State Correction (kBT): ", dG_std_corr)
println("Solvation Free Energy (kBT):     ", dG)
println("Jackknife SE, no std-state uncertainty (kBT): ", dG_se)
println("=========================================")

##

beta = tss_state_solv.state_space.betas[1]

println("=========================================")
println("Annihilation in solvent (kJ mol^-1):   ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):    ", dG_vac / beta)
println("Standard State Correction (kJ mol^-1): ", dG_std_corr / beta)
println("Solvation Free Energy (kJ mol^-1):     ", dG / beta)
println("Jackknife SE, no std-state uncertainty (kJ mol^-1): ", dG_se / beta)
println("=========================================")

#=
=========================================
Annihilation in solvent (kBT):   8.696862
Annihilation in vacuum (kBT):    3.4346604
Standard State Correction (kBT): 3.249398594044294
Solvation Free Energy (kBT):     -2.012803191996966
Jackknife SE, no std-state uncertainty (kBT): 0.072397865
=========================================
=========================================
Annihilation in solvent (kJ mol^-1):   22.416018
Annihilation in vacuum (kJ mol^-1):    8.85278
Standard State Correction (kJ mol^-1): 8.375271054356045
Solvation Free Energy (kJ mol^-1):     -5.187966888071426
Jackknife SE, no std-state uncertainty (kJ mol^-1): 0.1866043
=========================================
=#
