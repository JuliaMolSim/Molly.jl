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
OUTPUT_PREFIX = "awh_solvation"

N_LAMBDA_STATES = 20
N_MD_STEPS = 50
AWH_TIME = FT(15)u"ns"
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

function setup_alchemical_awh(pdb_file, solute_indices; is_vacuum=false, rng=Random.default_rng())
    nonbonded_method = is_vacuum ? :none : :pme
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : FT(1) * u"nm"
    dist_buffer = is_vacuum ? FT(0) * u"nm" : FT(0.2) * u"nm"
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing
    awh_loggers = awh_solvation_loggers(is_vacuum)

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
                # Only update the global lambda; the scheduler handles the component logic.
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

solute_idx = 1:9

awh_state_solv, awh_sim_solv = setup_alchemical_awh(
    joinpath(data_dir, "ethanol_solv.pdb"),
    solute_idx;
    is_vacuum=false,
    rng=MersenneTwister(RNG_SEED),
)
println()

##
awh_state_vac, awh_sim_vac = setup_alchemical_awh(
    joinpath(data_dir, "ethanol_vac.pdb"),
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

dG_solv = f_solv[end] - f_solv[1]
dG_vac  = f_vac[end]  - f_vac[1]

# Standard gas state (1 bar) volume per molecule at T0
V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)

# Standard solution state (1 M = 1 mol/L) volume per molecule
V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))

# Analytical standard-state correction
dG_std_corr = log(V_gas / V_std)

# Final solvation free energy from the two-leg cycle
dG = dG_vac - dG_solv + dG_std_corr

##
println("=========================================")
println("Annihilation in solvent (kBT):   ", dG_solv)
println("Annihilation in vacuum (kBT):    ", dG_vac)
println("Standard State Correction (kBT): ", dG_std_corr)
println("Solvation Free Energy (kBT):     ", dG)
println("=========================================")

##

beta = awh_state_solv.state_space.betas[1]

println("=========================================")
println("Annihilation in solvent (kJ mol^-1):   ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):    ", dG_vac / beta)
println("Standard State Correction (kJ mol^-1): ", dG_std_corr / beta)
println("Solvation Free Energy (kJ mol^-1):     ", dG / beta)
println("=========================================")

#=
=========================================
Annihilation in solvent (kBT):   8.642657
Annihilation in vacuum (kBT):    3.4234502
Standard State Correction (kBT): 3.249398594044294
Solvation Free Energy (kBT):     -1.9698082159532646
=========================================
=========================================
Annihilation in solvent (kJ mol^-1):   22.276306
Annihilation in vacuum (kJ mol^-1):    8.823886
Standard State Correction (kJ mol^-1): 8.375271054356045
Solvation Free Energy (kJ mol^-1):     -5.077148049471093
=========================================
=#
