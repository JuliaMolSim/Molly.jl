using Molly
using Test

temperature = 298

# Ideal gas 2D
timestep = 0.002
n_steps = 1000
n_atoms = 20
box_size = 2.0

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=[Atom("Ar", "Ar", i, "Ar", 0.0, 10.0, 0.3, 0.2) for i in 1:n_atoms],
    general_inters=Dict("LJ" => LennardJones(true)),
    coords=[box_size .* rand(SVector{2}) for i in 1:n_atoms],
    velocities=[velocity(10.0, temperature, dims=2) .* 0.01 for i in 1:n_atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
    thermostat=AndersenThermostat(10.0),
    loggers=Dict("temp" => TemperatureLogger(100),
                    "coords" => CoordinateLogger(100, dims=2)),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s, parallel=false)

# Ideal gas 3D
n_atoms = 100

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=[Atom("Ar", "Ar", i, "Ar", 0.0, 10.0, 0.3, 0.2) for i in 1:n_atoms],
    general_inters=Dict("LJ" => LennardJones(true)),
    coords=[box_size .* rand(SVector{3}) for i in 1:n_atoms],
    velocities=[velocity(10.0, temperature) .* 0.01 for i in 1:n_atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
    thermostat=AndersenThermostat(10.0),
    loggers=Dict("temp" => TemperatureLogger(100),
                    "coords" => CoordinateLogger(100)),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s, parallel=false)

final_coords = s.loggers["coords"].coords[end]
displacements(final_coords, box_size)
distances(final_coords, box_size)
rdf(final_coords, box_size)

# Molecules
coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]
for i in 1:length(coords)
    push!(coords, coords[i] .+ [0.1, 0.0, 0.0])
end
bonds = [Bond(i, Int(i + n_atoms / 2), 0.1, 300_000) for i in 1:(n_atoms / 2)]

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=[Atom("H", "H", i, "H", 0.0, 10.0, 0.3, 0.2) for i in 1:n_atoms],
    specific_inter_lists=Dict("Bonds" => bonds),
    general_inters=Dict("LJ" => LennardJones(true)),
    coords=coords,
    velocities=[velocity(10.0, temperature) .* 0.01 for i in 1:n_atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
    thermostat=AndersenThermostat(10.0),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "coords" => CoordinateLogger(10)),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s, parallel=false)

# Protein
timestep = 0.0002
n_steps = 100
atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            normpath(@__DIR__, "..", "data", "5XER", "gmx_top_ff.top"),
            normpath(@__DIR__, "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temperature) .* 0.01 for a in atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(10.0),
    loggers=Dict("temp" => TemperatureLogger(10)),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s, parallel=false)
