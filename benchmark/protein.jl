using Molly
using DelimitedFiles

data_dir = normpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")
openmm_dir = joinpath(data_dir, "openmm_6mrr")

ff = OpenMMForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
velocities = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"
s = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; velocities=velocities)
simulator = VelocityVerlet(dt=0.0005u"ps", coupling=AndersenThermostat(300.0u"K", 1.0u"ps"))
n_steps = 500

parallel = false
simulate!(s, simulator, 5; parallel=parallel)
@time simulate!(s, simulator, n_steps; parallel=parallel)

parallel = true
simulate!(s, simulator, 5; parallel=parallel)
@time simulate!(s, simulator, n_steps; parallel=parallel)
