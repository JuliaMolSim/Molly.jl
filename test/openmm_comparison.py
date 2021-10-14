# Calculate energy and forces with OpenMM for comparison to Molly
# Used OpenMM v7.6.0, Python v3.9.5

from openmm.app import *
from openmm import *
from openmm.unit import *
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
ff_dir = os.path.join(data_dir, "force_fields")
out_dir = os.path.join(data_dir, "openmm_6mrr")
pdb_file = os.path.join(data_dir, "6mrr_equil.pdb")

platform = Platform.getPlatformByName("Reference")

for inter in ["bond", "angle", "proptor", "improptor", "lj", "coul", "all"]:
    pdb = PDBFile(pdb_file)
    if inter == "all":
        force_field = ForceField(
            os.path.join(ff_dir, f"ff99SBildn.xml"),
            os.path.join(ff_dir, f"tip3p_standard.xml"),
        )
    else:
        force_field = ForceField(
            os.path.join(ff_dir, f"ff99SBildn_{inter}_only.xml"),
            os.path.join(ff_dir, f"tip3p_standard_{inter}_only.xml"),
        )

    system = force_field.createSystem(pdb.topology, nonbondedMethod=CutoffPeriodic,
                                        nonbondedCutoff=1*nanometer, constraints=None,
                                        rigidWater=False, removeCMMotion=False,
                                        switchDistance=None, useDispersionCorrection=False)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()

    with open(os.path.join(out_dir, f"forces_{inter}_only.txt"), "w") as of:
        for force in forces:
            of.write(f"{force.x} {force.y} {force.z}\n")

    with open(os.path.join(out_dir, f"energy_{inter}_only.txt"), "w") as of:
        of.write(f"{energy.value_in_unit(energy.unit)}\n")
