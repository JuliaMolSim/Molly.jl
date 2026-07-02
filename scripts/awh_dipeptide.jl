##
using Molly
using CUDA
using GLMakie
using Random

CUDA.device!(parse(Int, get(ENV, "MOLLY_CUDA_DEVICE", "0")))

##
FT = Float32
AT = CuArray

RNG_SEED = 42
OUTPUT_PREFIX = "awh_dipeptide"
rng = MersenneTwister(RNG_SEED)

DT = FT(4)u"fs"

TIME_EQ = FT(100)u"ps"
STEPS_EQ = Int(floor(TIME_EQ/DT))

TEMP = FT(310)u"K"
TAU_T = FT(0.1)u"ps"

thermostat = VelocityRescaleThermostat(TEMP, TAU_T; n_steps = 1)

PRES = one(FT)u"bar"
TAU_P = FT(4)u"ps"

N_PRES = Int(floor(FT(0.1 * TAU_P / DT)))

barostat = CRescaleBarostat(PRES, TAU_P; n_steps = N_PRES)

##

data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")
ff = MolecularForceField(
    FT,
    joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
    units=true)

sys = System(
    joinpath(data_dir, "..", "exercises", "dipeptide_equil.pdb"),
    ff;
    array_type=AT,
    nonbonded_method=:pme,
    constraints=:hbonds,
    rigid_water = true,
    hydrogen_mass=3
)

##

minim = SteepestDescentMinimizer(max_steps = 5000)
simulate!(sys, minim)

random_velocities!(sys, TEMP; rng = rng)

##

vverlet = VelocityVerlet(DT, (thermostat, barostat), 100)
simulate!(sys, vverlet, STEPS_EQ)

##

PHI_INDS = [5, 7, 9, 15]
PSI_INDS = [7, 9, 15, 17]
PHI_CV = CalcTorsion(PHI_INDS, :pbc, true)
PSI_CV = CalcTorsion(PSI_INDS, :pbc, true)

N_PHI_STATES = 20
N_PSI_STATES = 20

PHI_MIN = FT(-π)
PHI_MAX = FT(π)
PSI_MIN = FT(-π)
PSI_MAX = FT(π)
FLAT_BOTTOM_WIDTH = ustrip(u"rad", FT(360 / N_PHI_STATES)u"°")
BIAS_K = FT(100.0)u"kJ * mol^-1"

PHI_TARGETS = collect(range(PHI_MIN, PHI_MAX; length=N_PHI_STATES + 1))[1:end-1]
PSI_TARGETS = collect(range(PSI_MIN, PSI_MAX; length=N_PSI_STATES + 1))[1:end-1]

##

thermo_states = ThermoState[]

# Match the column-major state indexing used by tss_grid_graph((N_PHI, N_PSI)).
for psi in PSI_TARGETS
    for phi in PHI_TARGETS

        bias_phi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, phi)
        bias_psi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, psi)

        bp_phi = BiasPotential(PHI_CV, bias_phi)
        bp_psi = BiasPotential(PSI_CV, bias_psi)

        sys_bias = System(deepcopy(sys),
            general_inters=(sys.general_inters..., bp_phi, bp_psi),
        )

        push!(thermo_states, ThermoState(sys_bias, vverlet))

    end
end

##

awh_state = AWHState(
    thermo_states;
    first_state=1,
    n_bias=100,
    reuse_neighbors=true,
)

N_MD_STEPS = 50

AWH_TIME = FT(25)u"ns"
TOTAL_STEPS = Int(floor(AWH_TIME / DT))

function save_state_histogram(states, bins, path)
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

    hist!(
        ax,
        states;
        bins = bins,
        color = :royalblue,
        alpha = 0.35,
        strokewidth = 1,
        strokecolor = :black,
        normalization = :pdf,
        label = "AWH",
    )

    axislegend(position = :rt, labelsize = 20)
    display(fig)
    save(path, fig)
end

awh_visited_states(state) = collect(state.stats.active_λ)

pmf_deconv = PMFDeconvolution(
    awh_state;
    grid = ((PHI_MIN, PSI_MIN), (PHI_MAX, PSI_MAX), (N_PHI_STATES, N_PSI_STATES)),
)

awh_sim = AWHSimulation(
    awh_state;
    num_md_steps = N_MD_STEPS,
    update_freq = 1,
    well_tempered_factor = FT(500.0),
    log_freq = 10,
    loggers = (traj = TrajectoryWriter(1000, "$(OUTPUT_PREFIX).dcd"),),
    pmf = pmf_deconv,
)

##
Random.seed!(RNG_SEED + 1)
simulate!(awh_sim, TOTAL_STEPS)

##
pmf_result = pmf(awh_sim.pmf)
pmf_kbt = pmf_result.F
pmf_plot = map(x -> isfinite(x) ? x : FT(NaN), pmf_kbt)

##

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
          title = L"\textbf{Free Energy}",
          xlabel = L"\textbf{\phi / rad}",
          ylabel = L"\textbf{\psi / rad}",
          xlabelsize = 20, ylabelsize = 20,
          titlesize = 24,
          xlabelfont = :bold, ylabelfont = :bold,
          xticklabelsize = 18, yticklabelsize = 18)

hm_fe = heatmap!(
    ax_fe,
    PHI_TARGETS,
    PSI_TARGETS,
    pmf_plot;
    colormap = :inferno,
    nan_color = (:gray, 0.35),
)

Colorbar(
    fig_fe[1, 2],
    hm_fe;
    label = L"\textbf{F / k_{B}T}",
    labelsize = 20,
    ticklabelsize = 16,
)

ax_fe.aspect = DataAspect()
display(fig_fe)

save("$(OUTPUT_PREFIX)_pmf.png", fig_fe)

##

deltaF = awh_state.stats.max_delta_f_history
iter = awh_state.stats.step_indices

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
    iter,
    log10.(deltaF);
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "PHI/PSI AWH"
)

axislegend(
    position = :rt,
    labelsize = 24
)

display(fig_df)

save("$(OUTPUT_PREFIX)_convergence.png", fig_df)

##
state_bins = 0.5:1:(N_PHI_STATES * N_PSI_STATES + 0.5)
save_state_histogram(
    awh_visited_states(awh_state),
    state_bins,
    "$(OUTPUT_PREFIX)_states.png",
)
