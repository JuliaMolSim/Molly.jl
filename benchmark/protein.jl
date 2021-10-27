using Molly
using DelimitedFiles

data_dir = normpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")
openmm_dir = joinpath(data_dir, "openmm_6mrr")

ff = OpenMMForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)

atoms, atoms_data, specific_inter_lists, general_inters, neighbor_finder, coords, box_size = setupsystem(
    joinpath(data_dir, "6mrr_equil.pdb"), ff)

n_steps = 500
timestep = 0.0005u"ps"
temp = 300.0u"K"
velocities = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    atoms_data=atoms_data,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temp,
    box_size=box_size,
    neighbor_finder=neighbor_finder,
    thermostat=AndersenThermostat(1.0u"ps"),
    timestep=timestep,
    n_steps=n_steps,
)

parallel = false
simulate!(s, 5; parallel=parallel)
@time simulate!(s, n_steps; parallel=parallel)

parallel = true
simulate!(s, 5; parallel=parallel)
@time simulate!(s, n_steps; parallel=parallel)
