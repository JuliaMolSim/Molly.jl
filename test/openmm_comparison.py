# Calculate energy and forces with OpenMM for comparison to Molly
# Used OpenMM v8.2.0, Python v3.11.12

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

class VelocityVerletIntegrator(CustomIntegrator):
    def __init__(self, time_step):
        super(VelocityVerletIntegrator, self).__init__(time_step)
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

inters = [
    "bond_only", "angle_only", "proptor_only", "improptor_only", "lj_only", "coul_only",
    "all_cut", "all_pme",
]

for inter in inters:
    pdb = PDBFile(pdb_file)
    if inter.startswith("all"):
        force_field = ForceField(
            os.path.join(ff_dir, f"ff99SBildn.xml"),
            os.path.join(ff_dir, f"tip3p_standard.xml"),
        )
    else:
        force_field = ForceField(
            os.path.join(ff_dir, f"ff99SBildn_{inter}.xml"),
            os.path.join(ff_dir, f"tip3p_standard_{inter}.xml"),
        )
    nonbondedMethod = PME if inter == "all_pme" else CutoffPeriodic

    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=1*nanometer,
        constraints=None,
        rigidWater=False,
        switchDistance=None,
        useDispersionCorrection=False,
    )
    integrator = VelocityVerletIntegrator(time_step)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()

    with open(os.path.join(out_dir, f"forces_{inter}.txt"), "w") as of:
        for force in forces:
            of.write(f"{force.x} {force.y} {force.z}\n")

    with open(os.path.join(out_dir, f"energy_{inter}.txt"), "w") as of:
        of.write(f"{energy.value_in_unit(energy.unit)}\n")

    # Run a short simulation with all interactions
    if inter == "all_cut":
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
