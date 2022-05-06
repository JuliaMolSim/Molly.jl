using Molly

n_atoms = 100
atom_mass = 10.0u"u"
atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
box_size = SVector(2.0, 2.0, 2.0)u"nm"

coords = place_atoms(n_atoms ÷ 2, box_size, 0.3u"nm")
for i in 1:length(coords)
    push!(coords, coords[i] .+ [0.1, 0.0, 0.0]u"nm")
end

temp = 100.0u"K"
velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]

bonds = InteractionList2Atoms(
    collect(1:(n_atoms ÷ 2)),           # First atom indices
    collect((1 + n_atoms ÷ 2):n_atoms), # Second atom indices
    repeat([""], n_atoms ÷ 2),          # Bond types
    [HarmonicBond(b0=0.1u"nm", kb=300_000.0u"kJ * mol^-1 * nm^-2") for i in 1:(n_atoms ÷ 2)],
)

specific_inter_lists = (bonds,)

# All pairs apart from bonded pairs are eligible for non-bonded interactions
nb_matrix = trues(n_atoms, n_atoms)
for i in 1:(n_atoms ÷ 2)
    nb_matrix[i, i + (n_atoms ÷ 2)] = false
    nb_matrix[i + (n_atoms ÷ 2), i] = false
end

neighbor_finder = DistanceNeighborFinder(
    nb_matrix=nb_matrix,
    n_steps=10,
    dist_cutoff=1.5u"nm",
)

sys = System(
    atoms=atoms,
    pairwise_inters=(LennardJones(nl_only=true),),
    specific_inter_lists=specific_inter_lists,
    coords=coords,
    velocities=velocities,
    box_size=box_size,
    neighbor_finder=neighbor_finder,
    loggers=Dict(
        "temp" => TemperatureLogger(10),
        "coords" => CoordinateLogger(10),
    ),
)

simulator = VelocityVerlet(
    dt=0.002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)
simulate!(sys, simulator, 1_000)

using GLMakie
visualize(
    sys.loggers["coords"],
    box_size,
    "sim_diatomic.mp4";
    connections=[(i, i + (n_atoms ÷ 2)) for i in 1:(n_atoms ÷ 2)],
)
