using Molly
using GLMakie

atoms = [Atom(mass=1.0f0), Atom(mass=1.0f0)]
coords = [SVector(0.3f0, 0.5f0), SVector(0.7f0, 0.5f0)]
velocities = [SVector(0.0f0, 1.0f0), SVector(0.0f0, -1.0f0)]
pairwise_inters = (Gravity(nl_only=false, G=-1.5f0),)
simulator = VelocityVerlet(dt=0.002f0)
box_size = SVector(1.0f0, 1.0f0)

sys = System(
    atoms=atoms,
    pairwise_inters=pairwise_inters,
    coords=coords,
    velocities=velocities,
    box_size=box_size,
    loggers=Dict("coords" => CoordinateLogger(Float32, 10; dims=2)),
    force_units=NoUnits,
    energy_units=NoUnits,
)

simulate!(sys, simulator, 2_000)

Molly.visualize(
    sys.loggers["coords"],
    box_size,
    "sim_gravity.mp4";
    trails=4,
    framerate=15,
    color=[:orange, :lightgreen],
)
