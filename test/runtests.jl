using Molly
using Base.Test

max_starting_velocity = 0.1
timestep = 0.0002
n_steps = 50

forcefield, molecule, coords, box_size = readinputs(
            normpath(@__DIR__, "..", "data", "5XER", "gmx_top_ff.top"),
            normpath(@__DIR__, "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(forcefield, molecule, coords, box_size,
            max_starting_velocity, timestep, n_steps)

simulate!(s)

writepdb("test.pdb", s.universe)
rm("test.pdb")
