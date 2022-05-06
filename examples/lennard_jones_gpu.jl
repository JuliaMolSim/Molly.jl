using Molly
using CUDA

n_atoms = 100
atom_mass = 10.0f0u"u"
box_size = SVector(2.0f0, 2.0f0, 2.0f0)u"nm"
temp = 100.0f0u"K"
atoms = cu([Atom(mass=atom_mass, σ=0.3f0u"nm", ϵ=0.2f0u"kJ * mol^-1") for i in 1:n_atoms])
coords = cu(place_atoms(n_atoms, box_size, 0.3u"nm"))
velocities = cu([velocity(atom_mass, temp) for i in 1:n_atoms])
simulator = VelocityVerlet(dt=0.002f0u"ps")

sys = System(
    atoms=atoms,
    pairwise_inters=(LennardJones(),),
    coords=coords,
    velocities=velocities,
    box_size=box_size,
    loggers=Dict(
        "temp"   => TemperatureLogger(typeof(1.0f0u"K"), 10),
        "coords" => CoordinateLogger(typeof(1.0f0u"nm"), 10),
    ),
)

simulate!(sys, simulator, 1_000)

# visualization
using GLMakie
visualize(sys.loggers["coords"], box_size, "sim_lj_gpu.mp4")

