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
Δt = FT(2)u"fs"
T0 = FT(310)u"K"
P0 = FT(1)u"bar"
RNG_SEED = 20240520

N_LAMBDA_STATES = 20
N_MD_STEPS = 50
AWH_TIME = FT(4)u"ns"
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"

# Annihilate by running from full interactions at λ=1 to decoupled at λ=0.
# InsertRole with DefaultLambdaScheduler keeps sterics fully on while
# electrostatics are removed from λ=1 -> 0.5, then removes sterics from
# λ=0.5 -> 0.
lambda_schedule = FT.(range(1.0, stop=0.0, length=N_LAMBDA_STATES))

# --- Force Field Setup ---
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")
ff = MolecularForceField(FT, joinpath.(ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)

##
function alchemical_coulomb_softcore(coul::CoulombEwald)
    return CoulombSoftCoreGapsysEwald(
        dist_cutoff = coul.dist_cutoff,
        error_tol = coul.error_tol,
        α = FT(0.3),
        σQ = FT(1.0)u"nm",
        use_neighbors = coul.use_neighbors,
        scheduler = Molly.DefaultLambdaScheduler(),
        weight_special = coul.weight_special,
        coulomb_const = coul.coulomb_const,
        approximate_erfc = coul.approximate_erfc,
    )
end

function alchemical_coulomb_softcore(coul::Coulomb)
    return CoulombSoftCoreGapsys(
        cutoff = coul.cutoff,
        α = FT(0.3),
        σQ = FT(1.0)u"nm",
        use_neighbors = coul.use_neighbors,
        scheduler = Molly.DefaultLambdaScheduler(),
        weight_special = coul.weight_special,
        coulomb_const = coul.coulomb_const,
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

function setup_alchemical_awh(pdb_file, solute_indices; is_vacuum=false, rng=Random.default_rng())
    nonbonded_method = is_vacuum ? :none : :pme
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : FT(1) * u"nm"
    dist_buffer = is_vacuum ? FT(0) * u"nm" : FT(0.2) * u"nm"
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing

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
        hydrogen_mass=2
    )

    thermostat = VelocityRescaleThermostat(T0, FT(0.1)u"ps"; n_steps=1)

    if is_vacuum
        integrator = VelocityVerlet(Δt, (thermostat,), 100)
    else
        barostat = CRescaleBarostat(P0, FT(4)u"ps"; n_steps=200)
        integrator = VelocityVerlet(Δt, (thermostat, barostat), 100)
    end

    minim = SteepestDescentMinimizer(step_size=FT(0.01)u"nm", max_steps=1000)
    simulate!(sys_base, minim)
    random_velocities!(sys_base, T0; rng=rng)

    equil_time = is_vacuum ? VACUUM_EQUIL_TIME : SOLVENT_EQUIL_TIME
    equil_steps = Int(floor(equil_time / Δt))
    simulate!(sys_base, integrator, equil_steps; rng=rng)

    p_inters = sys_base.pairwise_inters
    idx_lj   = findfirst(x -> x isa LennardJones, p_inters)
    idx_coul = findfirst(x -> x isa Union{Coulomb, CoulombEwald}, p_inters)
    isnothing(idx_lj) && error("could not find LennardJones pairwise interaction")
    isnothing(idx_coul) && error("could not find Coulomb or CoulombEwald pairwise interaction")

    lj_sc = alchemical_lj_softcore(p_inters[idx_lj])
    cl_sc = alchemical_coulomb_softcore(p_inters[idx_coul])

    atoms_cpu = Molly.from_device(sys_base.atoms)
    thermo_states = ThermoState[]

    for λ in lambda_schedule
        acopy = Atom[]
        for a in atoms_cpu
            if a.index ∈ solute_indices
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(λ), Molly.InsertRole))
            else
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(a.λ), a.alch_role))
            end
        end

        sys_w = System(
            deepcopy(sys_base);
            atoms = Molly.to_device([acopy...], AT),
            pairwise_inters = (lj_sc, cl_sc)
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

save("awh_solv_profile.png", fig_fe)

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

save("awh_solv_convergence.png", fig_df)

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
