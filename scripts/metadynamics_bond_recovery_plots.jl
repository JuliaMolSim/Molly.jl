##
# Compares standard (constant-height) Metadynamics against well-tempered Metadynamics by
# looking at what each actually samples: a histogram of the visited bond distance over the
# run. Metadynamics is an enhanced-sampling method, so both biased histograms come out
# broader than the unbiased equilibrium distribution -- and the untempered run, having
# flooded the well, broader still than the well-tempered run.
#
# The biased histograms can't be compared with the unbiased distribution directly. The
# second figure reweights the well-tempered samples from the second half of the run, where
# the bias changes slowly, by exp(V_final(r) / kT) with the final bias V_final, which
# recovers the unbiased distribution r^2 exp(-U(r) / kT).
#
# Runs the same setup as the "MetaDynamics simulation" test in test/bias.jl, once with
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
grid_min, grid_max, n_bins = 0.3, 1.7, 701

bias_factor = 10.0 # gamma: well-tempered "temperature boost", must be > 1
n_steps = 500_000
deposit_interval = 100
record_every = 10 # how often to log the sampled distance, for the histogram

##
function run_metadynamics(tempering; label="")
    atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2), Atom(mass=mass, σ=0.3, ϵ=0.2)]
    coords = [SVector(4.5, 5.0, 5.0), SVector(4.5 + r0, 5.0, 5.0)]
    velocities = [random_velocity(mass, temp) for i in 1:2]

    memory = GridHills((calc_dist,), 0.2, 0.02, grid_min, grid_max, n_bins)
    # deposit_interval paces deposits by simulation step from within forces! (called every step regardless of
    # simulator), so no external logger is needed to drive hill deposition
    bias = MetaDynamics(memory; deposit_interval=deposit_interval, tempering=tempering)

    # This logger is purely for recording the sampled distance for the histogram below --
    # unrelated to hill deposition, which bias paces itself via deposit_interval
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
    @time simulate!(sys, simulator, n_steps)
    n_deposits = bias.n_deposits[]
    println("  $(n_deposits) hills deposited onto a $(n_bins)-point grid over [$(grid_min), $(grid_max)]")

    return bias, values(sys.loggers.dist)
end

bias_untempered, dist_untempered = run_metadynamics(NoTempering(); label="Untempered")
bias_tempered, dist_tempered = run_metadynamics(WellTemperedTempering(bias_factor, kT); label="Well-tempered")

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
# Reweight the well-tempered samples from the second half of the run, where the bias is
# close to converged, by exp(β V_final(r)) to undo the bias
dist_late = in_window(dist_tempered[(end ÷ 2 + 1):end])
weights_late = [exp(β * potential_energy(bias_tempered.memory, r)) for r in dist_late]

fig_rw = histogram(
    dist_late;
    weights = weights_late,
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
