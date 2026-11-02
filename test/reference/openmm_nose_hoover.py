# Run the Nosé-Hoover chain integrator with OpenMM for comparison to Molly
# The starting coordinates and velocities are generated from formulae that are repeated
#   in the Molly test, the values printed at the end are hard-coded there
# Used OpenMM v8.4.0, Python v3.11.14
# An output directory can be given as an argument to write all the final coordinates
#   and velocities

from openmm import *
from openmm.unit import *
import math
import sys

platform = Platform.getPlatformByName("Reference")
n_steps = 200
dt = 0.002 # ps
dist_cutoff = 0.9 # nm
box_size = 2.0 # nm
chain_length, n_mts, n_ys = 3, 3, 7

def run_system(label, masses, positions, velocities, σ, ϵ, bonded_pairs, temperature,
               collision_frequency):
    n_atoms = len(masses)
    system = System()
    system.setDefaultPeriodicBoxVectors(Vec3(box_size, 0, 0), Vec3(0, box_size, 0),
                                        Vec3(0, 0, box_size))
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(dist_cutoff)
    nonbonded.setUseDispersionCorrection(False)
    nonbonded.setUseSwitchingFunction(False)
    for m in masses:
        system.addParticle(m)
        nonbonded.addParticle(0.0, σ, ϵ)
    for (i, j, d) in bonded_pairs:
        system.addConstraint(i, j, d)
        nonbonded.addException(i, j, 0.0, 1.0, 0.0)
    system.addForce(nonbonded)
    system.addForce(CMMotionRemover(1))

    integrator = NoseHooverIntegrator(temperature, collision_frequency, dt, chain_length,
                                      n_mts, n_ys)
    integrator.setConstraintTolerance(1e-10)
    context = Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocities(velocities)
    integrator.step(n_steps)

    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                             enforcePeriodicBox=True)
    coords = state.getPositions(asNumpy=False).value_in_unit(nanometer)
    vels = state.getVelocities(asNumpy=False).value_in_unit(nanometer / picosecond)
    ke = state.getKineticEnergy().value_in_unit(kilojoule_per_mole)
    pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    hb = integrator.computeHeatBathEnergy().value_in_unit(kilojoule_per_mole)
    n_dof = integrator.getThermostat(0).getNumDegreesOfFreedom()
    print(f"{label}: n_dof {n_dof}")
    print(f"    kinetic energy    {ke!r}")
    print(f"    potential energy  {pe!r}")
    print(f"    heat bath energy  {hb!r}")
    for i in (0, 1):
        print(f"    coords {i + 1}  {[coords[i][d] for d in range(3)]!r}")
        print(f"    velocities {i + 1}  {[vels[i][d] for d in range(3)]!r}")
    if len(sys.argv) > 1:
        with open(f"{sys.argv[1]}/{label}_coords.txt", "w") as of:
            for c in coords:
                of.write(f"{c[0]!r} {c[1]!r} {c[2]!r}\n")
        with open(f"{sys.argv[1]}/{label}_velocities.txt", "w") as of:
            for v in vels:
                of.write(f"{v[0]!r} {v[1]!r} {v[2]!r}\n")
        with open(f"{sys.argv[1]}/{label}_energies.txt", "w") as of:
            of.write(f"{ke!r} {pe!r} {hb!r}\n")

def remove_mean(vecs):
    mean = Vec3(0, 0, 0)
    for v in vecs:
        mean += v
    mean = mean / len(vecs)
    return [v - mean for v in vecs]

# Lennard-Jones fluid on a perturbed lattice
n_side = 5
spacing = box_size / n_side
masses, positions, velocities = [], [], []
for i in range(n_side):
    for j in range(n_side):
        for k in range(n_side):
            n = n_side * n_side * i + n_side * j + k
            masses.append(39.948)
            positions.append(Vec3(spacing * i + 0.2 + 0.02 * math.sin(1.3 * n),
                                  spacing * j + 0.2 + 0.02 * math.cos(2.1 * n),
                                  spacing * k + 0.2 + 0.02 * math.sin(0.7 * n + 0.5)))
            velocities.append(Vec3(0.3 * math.cos(0.9 * n), 0.3 * math.sin(1.7 * n),
                                   0.3 * math.cos(2.3 * n)))
velocities = remove_mean(velocities)
run_system("lj", masses, positions, velocities, 0.34, 0.997, [], 120.0, 10.0)

# Diatomic molecules with constrained bonds
n_side = 4
spacing = box_size / n_side
bond_length = 0.15
masses, positions, velocities, bonded_pairs = [], [], [], []
for i in range(n_side):
    for j in range(n_side):
        for k in range(n_side):
            m = n_side * n_side * i + n_side * j + k
            u = Vec3(math.sin(1.1 * m + 0.3), math.cos(0.7 * m), math.sin(1.9 * m + 1.0))
            u = u / math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            centre = Vec3(spacing * i + 0.25, spacing * j + 0.25, spacing * k + 0.25)
            for s in (-1, 1):
                n = len(masses)
                masses.append(14.0)
                positions.append(centre + u * (s * bond_length / 2))
                velocities.append(Vec3(0.4 * math.cos(0.9 * n), 0.4 * math.sin(1.7 * n),
                                       0.4 * math.cos(2.3 * n)))
            bonded_pairs.append((len(masses) - 2, len(masses) - 1, bond_length))
velocities = remove_mean(velocities)
run_system("diatomic", masses, positions, velocities, 0.32, 0.3, bonded_pairs, 150.0, 20.0)
