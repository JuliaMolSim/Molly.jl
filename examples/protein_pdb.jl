using Molly

ff = OpenMMForceField("data/force_fields/ff99SBildn.xml",
                      "data/force_fields/tip3p_standard.xml",
                      "data/force_fields/his.xml")

sys = System(
    "data/6mrr_equil.pdb",
    ff;
    loggers=Dict(
        "energy" => TotalEnergyLogger(10),
        "writer" => StructureWriter(10, "traj_6mrr_1ps.pdb", ["HOH"]),
    ),
)

minimizer = SteepestDescentMinimizer()
simulate!(sys, minimizer)

random_velocities!(sys, 298.0u"K")
simulator = Langevin(
    dt=0.001u"ps",
    temperature=300.0u"K",
    friction=1.0u"ps^-1",
)

simulate!(sys, simulator, 5_000; parallel=true)
