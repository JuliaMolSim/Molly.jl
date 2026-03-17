##
using Revise
using Molly
using CUDA
using BenchmarkTools

data_dir = normpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")

AT = CuArray
T = Float32
ff = MolecularForceField(
    T,
    joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
    units=true,
)

sys = System(
    joinpath(data_dir, "6mrr_equil.pdb"),
    ff;
    units=true,
    array_type=AT,
    nonbonded_method=:none,
)

minim = SteepestDescentMinimizer(step_size = T(0.05)u"nm", max_steps = 5_000)

simulate!(sys, minim)

random_velocities!(sys, T(310)u"K")

sim = VelocityVerlet(dt = T(1)u"fs", coupling = (NoCoupling(),), remove_CM_motion = 100)

fs = forces(sys)

pe = potential_energy(sys) 

simulate!(sys, sim, 50)

@btime simulate!(sys, sim, 1_000)
