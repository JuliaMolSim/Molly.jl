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
vel_file = os.path.join(out_dir, "velocities_300K.txt")

platform = Platform.getPlatformByName("Reference")
n_steps = 100
time_step = 0.0005*picoseconds

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
    integrator = VerletIntegrator(time_step)
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

    # Run a short simulation with all interactions
    if inter == "all":
        if os.path.isfile(vel_file):
            # Load velocities if they already exist
            velocities = []
            with open(vel_file) as f:
                for line in f:
                    vel = [float(v) for v in line.rstrip().split()]
                    velocities.append(vel)
            simulation.context.setVelocities(velocities)
        else:
            # Generate consistent set of velocities for testing
            simulation.context.setVelocitiesToTemperature(300*kelvin)
            state = simulation.context.getState(getVelocities=True)
            velocities = state.getVelocities()
            with open(vel_file, "w") as of:
                for vel in velocities:
                    of.write(f"{vel.x} {vel.y} {vel.z}\n")

        simulation.step(n_steps)

        state = simulation.context.getState(getPositions=True, getVelocities=True)
        coords = state.getPositions()
        velocities = state.getVelocities()

        with open(os.path.join(out_dir, f"coordinates_{n_steps}steps.txt"), "w") as of:
            for coord in coords:
                of.write(f"{coord.x} {coord.y} {coord.z}\n")

        with open(os.path.join(out_dir, f"velocities_{n_steps}steps.txt"), "w") as of:
            for vel in velocities:
                of.write(f"{vel.x} {vel.y} {vel.z}\n")
