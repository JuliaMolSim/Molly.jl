using Molly
using Test

temperature = 298

# Test on an ideal gas
timestep = 0.002
n_steps = 1000
n_atoms = 100
box_size = 2.0

s = Simulation(
    VelocityVerlet(),
    [Atom("Ar", "Ar", i, "Ar", 0.0, 10.0, 0.3, 0.2) for i in 1:n_atoms],
    Dict(),
    Dict("LJ" => LennardJones(true)),
    [Coordinates(rand(3) .* box_size) for _ in 1:n_atoms],
    [Velocity(10.0, temperature) .* 0.01 for _ in 1:n_atoms],
    temperature,
    box_size,
    [],
    DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 4.0),
    AndersenThermostat(10.0),
    [TemperatureLogger(100), CoordinateLogger(100)],
    timestep,
    n_steps,
    0
)

simulate!(s)

# Test on molecules
coords = Coordinates[]
for _ in 1:(n_atoms / 2)
    c = rand(3) .* box_size
    push!(coords, Coordinates(c))
    push!(coords, Coordinates(c + [0.1, 0.0, 0.0]))
end
bonds = [Bond((i * 2) - 1, i * 2, 0.1, 300_000) for i in 1:(n_atoms / 2)]

s = Simulation(
    VelocityVerlet(),
    [Atom("H", "H", i, "H", 0.0, 10.0, 0.3, 0.2) for i in 1:n_atoms],
    Dict("Bonds" => bonds),
    Dict("LJ" => LennardJones(true)),
    coords,
    [Velocity(10.0, temperature) .* 0.01 for _ in 1:n_atoms],
    temperature,
    box_size,
    [],
    DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 4.0),
    AndersenThermostat(10.0),
    [TemperatureLogger(10), CoordinateLogger(10)],
    timestep,
    n_steps,
    0
)

simulate!(s)

# Test on a protein
timestep = 0.0002
n_steps = 100
atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            normpath(@__DIR__, "..", "data", "5XER", "gmx_top_ff.top"),
            normpath(@__DIR__, "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    VelocityVerlet(),
    atoms,
    specific_inter_lists,
    general_inters,
    coords,
    [Velocity(a.mass, temperature) .* 0.01 for a in atoms],
    temperature,
    box_size,
    [],
    DistanceNeighbourFinder(nb_matrix, 10),
    AndersenThermostat(10.0),
    [TemperatureLogger(10)],
    timestep,
    n_steps,
    0
)

simulate!(s)
