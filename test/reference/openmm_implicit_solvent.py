# Calculate implicit solvent energy and forces with OpenMM for comparison to Molly
# Used OpenMM commit a76c2de14b5a1ab604e95a5c4197e5a586e3000d, Python v3.9.7
# This version is required due to a carboxylate atom radius fix

from openmm.app import *
from openmm import *
from openmm.unit import *
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
ff_dir = os.path.join(data_dir, "force_fields")
out_dir = os.path.join(data_dir, "openmm_6mrr")
pdb_file = os.path.join(data_dir, "6mrr_nowater.pdb")
platform = Platform.getPlatformByName("Reference")

for solvent_model in ["obc2", "gbn2"]:
    pdb = PDBFile(pdb_file)
    force_field = ForceField(
        os.path.join(ff_dir, "ff99SBildn.xml"),
        f"implicit/{solvent_model}.xml",
    )
    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        implicitSolventKappa=1.0,
    )
    integrator = LangevinMiddleIntegrator(300*kelvin, 91/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()

    with open(os.path.join(out_dir, f"forces_{solvent_model}.txt"), "w") as of:
        for force in forces:
            of.write(f"{force.x} {force.y} {force.z}\n")

    with open(os.path.join(out_dir, f"energy_{solvent_model}.txt"), "w") as of:
        of.write(f"{energy.value_in_unit(energy.unit)}\n")
