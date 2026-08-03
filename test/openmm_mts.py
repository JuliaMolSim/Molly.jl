# Run multiple time step integration with OpenMM for comparison to Molly
# Used OpenMM v8.4.0, Python v3.11.14

from openmm.app import *
from openmm import *
from openmm.unit import *
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
out_dir = os.path.join(data_dir, "openmm_tip4pfb")

platform = Platform.getPlatformByName("Reference")
n_steps = 10
time_step = 0.001*picoseconds

pdb = PDBFile(os.path.join(data_dir, "tip4pew.pdb"))
forcefield = ForceField(os.path.join(data_dir, "force_fields", "tip4pfb.xml"))

for constraints in (None, HBonds):
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=1*nanometer,
        constraints=constraints,
        rigidWater=False,
        removeCMMotion=False,
    )

    for f in system.getForces():
        if isinstance(f, NonbondedForce):
            f.setForceGroup(0)
        elif isinstance(f, HarmonicAngleForce):
            f.setForceGroup(1)
        else: # HarmonicBondForce
            f.setForceGroup(2)

    if constraints == HBonds:
        cons_label = "cons"
        groups = [(0, 1), (1, 4)]
    else:
        cons_label = "nocons"
        groups = [(0, 1), (1, 4), (2, 8)]

    integrator = MTSIntegrator(time_step, groups)
    simulation = Simulation(pdb.topology, system, integrator, platform)

    simulation.context.setPositions(pdb.positions)
    simulation.context.computeVirtualSites()

    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()

    with open(os.path.join(out_dir, f"forces_{cons_label}.txt"), "w") as of:
        for force in forces:
            of.write(f"{force.x} {force.y} {force.z}\n")

    with open(os.path.join(out_dir, f"energy_{cons_label}.txt"), "w") as of:
        of.write(f"{energy.value_in_unit(energy.unit)}\n")

    simulation.step(n_steps)

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    coords = state.getPositions()
    velocities = state.getVelocities()

    with open(os.path.join(out_dir, f"coordinates_{n_steps}steps_{cons_label}.txt"), "w") as of:
        for coord in coords:
            of.write(f"{coord.x} {coord.y} {coord.z}\n")

    with open(os.path.join(out_dir, f"velocities_{n_steps}steps_{cons_label}.txt"), "w") as of:
        for vel in velocities:
            of.write(f"{vel.x} {vel.y} {vel.z}\n")
