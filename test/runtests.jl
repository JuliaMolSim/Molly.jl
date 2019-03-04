using Molly
using Test

timestep = 0.0002
temperature = 298
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
    [TemperatureLogger(100)],
    timestep,
    n_steps,
    0
)

simulate!(s)
