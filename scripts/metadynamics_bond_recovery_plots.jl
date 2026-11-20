##
# Compares standard (constant-height) Metadynamics against well-tempered Metadynamics by
# looking at what each actually samples: a histogram of the visited bond distance over the
# run. Metadynamics is an enhanced-sampling method, so both biased histograms come out
# broader than the unbiased equilibrium distribution -- and the untempered run, having
# flooded the well, broader still than the well-tempered run.
#
# The biased histograms can't be compared with the unbiased distribution directly. The
# second figure reweights the deposit frames of the well-tempered run with the weights of a
# MetadynamicsLogger, exp((V(s,t) - c(t)) / kT) (Tiwary & Parrinello, 2015), which recovers
# the unbiased distribution r^2 exp(-U(r) / kT).
#
# Runs the same setup as the "Metadynamics simulation" test in test/bias.jl, once with
# each tempering scheme.

using Molly
using Plots
using Unitful
using Random

##
mass = 10.0
r0 = 1.0
k_bond = 500.0
temp = 298.0
boundary = CubicBoundary(10.0)

bond = HarmonicBond(k=k_bond, r0=r0)
specific_inter_lists = (InteractionList2Atoms([1], [2], [bond]),)
calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)

kB = ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k) # Same convention System uses internally
kT = kB * temp
β = 1 / kT

# Grid spans r0 ± 0.7 nm, generous relative to the thermal spread sqrt(kT/k_bond) = ~0.07
# nm here, with bin_width finer than sigma so deposited hills are well resolved
grid_min, grid_max, n_points = 0.3, 1.7, 701

bias_factor = 10.0 # gamma: well-tempered "temperature boost", must be > 1
n_steps_run = 500_000
deposit_interval = 100 # Passed as n_steps to Metadynamics
record_every = 10 # how often to log the sampled distance, for the histogram

##
function run_metadynamics(tempering; label="")
    atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2), Atom(mass=mass, σ=0.3, ϵ=0.2)]
    coords = [SVector(4.5, 5.0, 5.0), SVector(4.5 + r0, 5.0, 5.0)]
    velocities = [random_velocity(mass, temp) for i in 1:2]

    memory = GridHills((calc_dist,), 0.2, 0.02, grid_min, grid_max, n_points)
    # n_steps paces deposits by simulation step from within forces! (called every step regardless of
    # simulator), so no external logger is needed to drive hill deposition
    # The temperature is given since the untempered run has none of its own
    bias = Metadynamics(memory; n_steps=deposit_interval, tempering=tempering,
                        logger=MetadynamicsLogger(temperature=temp))

    # This logger is purely for recording the sampled distance for the histogram below --
    # unrelated to hill deposition, which bias paces itself via n_steps
    dist_wrapper(sys, args...; kwargs...) = calculate_cv(calc_dist, sys.coords, sys.atoms, sys.boundary)

    # With only 2 atoms, removing the centre of mass velocity after each Andersen collision
    # would cool the bond to about 2/3 of temp
    simulator = VelocityVerlet(dt=0.002, coupling=AndersenThermostat(temp, 0.1),
                               remove_CM_motion=false)
    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        specific_inter_lists=specific_inter_lists,
        general_inters=(bias,),
        force_units=NoUnits,
        energy_units=NoUnits,
        loggers=(
            dist=GeneralObservableLogger(dist_wrapper, Float64, record_every),
        ),
    )

    print("$(label): ")
    @time simulate!(sys, simulator, n_steps_run)
    n_deposits = bias.n_deposits
    println("  $(n_deposits) hills deposited onto a $(n_points)-point grid over [$(grid_min), $(grid_max)]")

    return bias, values(sys.loggers.dist)
end

bias_untempered, dist_untempered = run_metadynamics(Molly.NoTempering(); label="Untempered")
bias_tempered, dist_tempered = run_metadynamics(WellTempered(bias_factor, temp); label="Well-tempered")

##
σ_thermal = sqrt(kT / k_bond)
xlims_plot = (r0 - 6 * σ_thermal, r0 + 6 * σ_thermal)

# Expected equilibrium density of an unbiased simulation: proportional to exp(-β U(r)),
# but since r is a radial distance in 3D (atom 2 free to move around atom 1 subject only
# to the isotropic bond potential), there's also an r^2 solid-angle Jacobian from
# marginalising out the angular coordinates
s_grid = collect(range(xlims_plot...; length=400))
bond_pe(r) = potential_energy(bond, SVector(0.0, 0.0, 0.0), SVector(r, 0.0, 0.0), boundary)
expected_density = s_grid.^2 .* exp.(-β .* bond_pe.(s_grid))
expected_density ./= sum(expected_density) * step(range(xlims_plot...; length=400)) # Normalise to a pdf over xlims_plot

##
gr() # Fast, no-display-required backend; swap for plotlyjs() if you want an interactive plot

# Only the portion of each recorded trajectory inside the plotted window, so the histogram
# pdf normalisation matches expected_density's normalisation over the same range
in_window(d) = filter(x -> xlims_plot[1] <= x <= xlims_plot[2], d)

# The biased sampling of both runs, broader than the unbiased distribution
fig = histogram(
    in_window(dist_untempered);
    label = "Untempered (biased sampling)",
    color = :royalblue,
    alpha = 1.0,
    normalize = :pdf,
    bins = range(xlims_plot...; length=60),
    title = "Sampled distance in the biased runs",
    xlabel = "Distance (nm)",
    ylabel = "Probability density",
    xlims = xlims_plot,
    legend = :topright,
    size = (900, 600),
)
histogram!(
    fig, in_window(dist_tempered);
    label = "Well-tempered (biased sampling)",
    color = :firebrick,
    alpha = 0.45,
    normalize = :pdf,
    bins = range(xlims_plot...; length=60),
)

display(fig)
savefig(fig, "./metadynamics_bond_recovery_plots.png")

##
# Reweight the deposit frames of the well-tempered run by their bias at sampling time
r_frames = values(bias_tempered.logger)
w_frames = Molly.metadynamics_weights(bias_tempered.logger)
keep = xlims_plot[1] .<= r_frames .<= xlims_plot[2]
dist_rw, weights_rw = r_frames[keep], w_frames[keep]

fig_rw = histogram(
    dist_rw;
    weights = weights_rw,
    label = "Well-tempered, reweighted",
    color = :firebrick,
    alpha = 0.6,
    normalize = :pdf,
    bins = range(xlims_plot...; length=60),
    title = "Reweighted distance vs. unbiased equilibrium distribution",
    xlabel = "Distance (nm)",
    ylabel = "Probability density",
    xlims = xlims_plot,
    legend = :topright,
    size = (900, 600),
)
plot!(
    fig_rw, s_grid, expected_density;
    label = "Expected equilibrium: r² exp(-β U(r))",
    color = :black,
    linewidth = 3,
)

display(fig_rw)
savefig(fig_rw, "./metadynamics_bond_recovery_reweighted.png")
