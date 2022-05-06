# New system with different file input
sys = System(
    joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"),
    joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top");
    loggers=Dict(
        "temp"   => TemperatureLogger(10),
        "writer" => StructureWriter(10, "traj_5XER_1ps.pdb"),
    ),
)

temp = 298.0u"K"
random_velocities!(sys, temp)
simulator = Verlet(
    dt=0.0002u"ps",
    coupling=BerendsenThermostat(temp, 1.0u"ps"),
)

simulate!(sys, simulator, 5_000)
