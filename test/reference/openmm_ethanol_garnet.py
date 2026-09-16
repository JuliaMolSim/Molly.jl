"""Reproduce the OpenMM reference energy for the solvated ethanol test system."""

from pathlib import Path
from xml.etree import ElementTree

import openmm
from openmm import Context, Platform, VerletIntegrator, app, unit


data_dir = Path(__file__).resolve().parent.parent / "data"
pdb = app.PDBFile(str(data_dir / "ethanol_garnet.pdb"))

templates = ElementTree.parse(
    data_dir / "force_fields" / "ethanol_garnet_residues.xml"
).getroot()
bonds = {
    residue.attrib["name"]: [
        (bond.attrib["from"], bond.attrib["to"]) for bond in residue.findall("Bond")
    ]
    for residue in templates.findall("Residue")
}
for residue in pdb.topology.residues():
    atoms = {atom.name: atom for atom in residue.atoms()}
    for atom_1, atom_2 in bonds[residue.name]:
        pdb.topology.addBond(atoms[atom_1], atoms[atom_2])

force_field = app.ForceField(str(data_dir / "force_fields" / "ethanol_garnet.xml"))
system = force_field.createSystem(
    pdb.topology,
    nonbondedMethod=app.CutoffPeriodic,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=None,
    rigidWater=False,
    removeCMMotion=False,
)
for group, force in enumerate(system.getForces()):
    force.setForceGroup(group)

platform = Platform.getPlatformByName("Reference")
integrator = VerletIntegrator(1.0 * unit.femtosecond)
context = Context(system, integrator, platform)
context.setPositions(pdb.positions)

energy_unit = unit.kilojoule_per_mole
energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
print(f"OpenMM {openmm.__version__}, {platform.getName()} platform")
print("CutoffPeriodic, cutoff=1.0 nm, constraints=None, rigidWater=False")
print(f"Potential energy: {energy:.15f} kJ/mol")
for group, force in enumerate(system.getForces()):
    component = context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
    print(f"{type(force).__name__}: {component.value_in_unit(energy_unit):.15f} kJ/mol")
