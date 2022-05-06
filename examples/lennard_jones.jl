using Molly

n_atoms = 100
atom_mass = 10.0u"u"
atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]

box_size = SVector(2.0, 2.0, 2.0)u"nm"
coords = place_atoms(n_atoms, box_size, 0.3u"nm") # Random placement without clashing

temp = 100.0u"K"
velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]

pairwise_inters = (LennardJones(),)

sys = System(
    atoms=atoms,
    pairwise_inters=pairwise_inters,
    coords=coords,
    velocities=velocities,
    box_size=box_size,
    loggers=Dict(
        "temp"   => TemperatureLogger(10),
        "coords" => CoordinateLogger(10),
    ),
)

simulator = VelocityVerlet(
    dt=0.002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)

simulate!(sys, simulator, 1_000)

# visualization
using GLMakie
visualize(sys.loggers["coords"], box_size, "sim_lj.mp4")
