# Calculate implicit solvent energy and forces with OpenMM for comparison to Molly
# Used OpenMM v7.7.0, Python v3.9.10

from openmm.app import *
from openmm import *
from openmm.unit import *
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
ff_dir = os.path.join(data_dir, "force_fields")
out_dir = os.path.join(data_dir, "openmm_6mrr")
pdb_file = os.path.join(data_dir, "6mrr_nowater.pdb")
platform = Platform.getPlatformByName("Reference")

pdb = PDBFile(pdb_file)
force_field = ForceField(os.path.join(ff_dir, "ff99SBildn.xml"), "implicit/obc2.xml")
system = force_field.createSystem(
    pdb.topology,
    nonbondedMethod=NoCutoff,
    constraints=None,
    rigidWater=False,
    removeCMMotion=False,
    switchDistance=None,
    useDispersionCorrection=False,
)
integrator = LangevinMiddleIntegrator(300*kelvin, 91/picosecond, 0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

state = simulation.context.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy()
forces = state.getForces()

with open(os.path.join(out_dir, "forces_obc2.txt"), "w") as of:
    for force in forces:
        of.write(f"{force.x} {force.y} {force.z}\n")

with open(os.path.join(out_dir, "energy_obc2.txt"), "w") as of:
    of.write(f"{energy.value_in_unit(energy.unit)}\n")
