import os
from openff.toolkit import Molecule
from openfe import SmallMoleculeComponent,SolventComponent, ProteinComponent
from openfe import ChemicalSystem
from openff.units import unit
from openfe.setup import LomapAtomMapper
from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score
from openfe.setup.ligand_network_planning import generate_lomap_network
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe import Transformation

import openmm
import numpy as np

def steric_core(x):
    return x

def elec_core(x):
    return x

def steric_insert(x):
    return 2.0 * x if x < 0.5 else 1.0

def steric_del(x):
    return 0.0 if x < 0.5 else 2.0 * (x - 0.5)

def elec_insert(x):
    return 0.0 if x < 0.5 else 2.0 * (x - 0.5)

def elec_del(x):
    return 2.0 * x if x < 0.5 else 1.0

def bond_core(x):
    return x

def angles_core(x):
    return x

def torsions_core(x):
    return x

global_parameters = {"lambda_bonds" : bond_core, "lambda_angles": angles_core, "lambda_torsions" : torsions_core,
                     "lambda_electrostatics_core" : elec_core,"lambda_sterics_core" : steric_core,"lambda_sterics_insert" : steric_insert,
                     "lambda_sterics_delete" : steric_del,"lambda_electrostatics_delete" : elec_del,
                     "lambda_electrostatics_insert": elec_insert}

def calc_energies_and_forces(openmm_system, openmm_positions, lammie):
    """
    """
    # Set λ to value provided in lammie
    for force in system.getForces():
        if hasattr(force, 'getNumGlobalParameters'):
            for i in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(i)
                if name in global_parameters:
                    force.setGlobalParameterDefaultValue(i, global_parameters[name](lammie))

    # Set forces to be in different groups
    for i, f in enumerate(openmm_system.getForces()):
        f.setForceGroup(i)

    f = openmm_system.getForces()[7]
    f.setReciprocalSpaceForceGroup(11)

    # Use the reference platform for testing
    platform = openmm.Platform.getPlatformByName('Reference')

    # Create Context
    integrator = openmm.openmm.VerletIntegrator(0.001*openmm.unit.picoseconds) 
    context = openmm.Context(openmm_system, integrator, platform)
    context.setPositions(openmm_positions)

    # Calculate energies for the different force groups of current state
    potential_energy_dict = {}
    for i, f in enumerate(openmm_system.getForces()):
        state = context.getState(getEnergy=True, groups={i})
        potential_energy_dict[f.getName()] = state.getPotentialEnergy()
    state = context.getState(getEnergy=True, groups={11})
    potential_energy_dict["PME"] = state.getPotentialEnergy()

    # Calculate forces for the different force groups of current state
    force_dict = {}
    for i, f in enumerate(openmm_system.getForces()):
        state = context.getState(getForces=True, groups={i})
        force_dict[f.getName()] = state.getForces(asNumpy=True)
    state = context.getState(getForces=True, groups={11})
    force_dict["PME"] = state.getForces(asNumpy=True)

    return potential_energy_dict, force_dict

#### TEST VARIABLES ####
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
ff_dir = os.path.join(data_dir, "force_fields")
out_dir = os.path.join(data_dir, "openmm_tyk2", "openfe")
os.chdir(out_dir)
pdb_file = os.path.join(data_dir, "tyk2_openmm.pdb")
lig_file = os.path.join(data_dir, "tyk2_ligands.sdf")

output_prefix = {"bond_only":["CustomBondForce","HarmonicBondForce"], "angle_only":["CustomAngleForce","HarmonicAngleForce"], 
                 "torsion_only":["CustomTorsionForce","PeriodicTorsionForce"], 
                 "nonbonded":["NonbondedForce","CustomNonbondedForce","CustomBondForce_exceptions"], "PME":["PME"], 
                 "all":["CustomBondForce","HarmonicBondForce","CustomAngleForce","HarmonicAngleForce",
                        "CustomTorsionForce","PeriodicTorsionForce","NonbondedForce","CustomNonbondedForce",
                        "CustomBondForce_exceptions","PME"]}

#### MAIN PIPELINE ####
# 1. Load ligands + create network + extract edge
# 2. Load Protein and define solvent
# 3. Creating chemical systems
# 4. Set settings
# 5. Create transformation
# 6. Run dry to extract alchemical system + save in pickle

### Can be rerun from here by re-loading pickle ###
# 7. Extract positions
# 8. Extract energies for λ=0
# 9. Extract energies for λ=1
# 10. Extract energies range of λ
########################

#### Step 1 ####
ligands_sdf = Molecule.from_file(lig_file)
ligand_mols = [SmallMoleculeComponent.from_openff(sdf) for sdf in ligands_sdf]
lomap_network = generate_lomap_network(
    ligands=ligand_mols,
    scorer=default_lomap_score,
    mappers=[LomapAtomMapper(element_change=False),])
mst_edges = [edge for edge in lomap_network.edges]
edge = mst_edges[0]

#### Step 2 ####
protein = ProteinComponent.from_pdb_file(pdb_file)
solvent = SolventComponent(positive_ion='Na', negative_ion='Cl',
                           neutralize=True, ion_concentration=0.15*unit.molar)

#### Step 3 ####
ejm_31_complex = ChemicalSystem({'ligand': edge.componentA,
                                  'solvent': solvent,
                                  'protein': protein,},
                               name=edge.componentA.name)
ejm_31_solvent = ChemicalSystem({'ligand': edge.componentA,
                                  'solvent': solvent,},
                               name=edge.componentA.name)

ejm_50_complex = ChemicalSystem({'ligand': edge.componentB,
                                 'solvent': solvent,
                                 'protein': protein,},
                               name=edge.componentB.name)
ejm_50_solvent = ChemicalSystem({'ligand': edge.componentB,
                                 'solvent': solvent,},
                               name=edge.componentB.name)

#### Step 4 ####
solvent_rbfe_settings = RelativeHybridTopologyProtocol.default_settings()
complex_rbfe_settings = RelativeHybridTopologyProtocol.default_settings()
complex_rbfe_settings.engine_settings.compute_platform = None
complex_rbfe_settings.forcefield_settings.constraints = None
complex_rbfe_settings.forcefield_settings.rigid_water = False
complex_rbfe_settings.forcefield_settings.hydrogen_mass = 1.007947
complex_rbfe_settings.integrator_settings.timestep = 1.0 * unit.femtosecond
complex_rbfe_settings.solvation_settings.box_shape = None
complex_rbfe_settings.solvation_settings.box_size = [8.0382, 8.0382, 8.0382] * unit.nanometers
complex_rbfe_settings.solvation_settings.solvent_padding = None

solvent_rbfe_protocol = RelativeHybridTopologyProtocol(
    settings=solvent_rbfe_settings
)
complex_rbfe_protocol = RelativeHybridTopologyProtocol(
    settings=complex_rbfe_settings
)

#### Step 5 ####
transformation_complex = Transformation(
            stateA=ejm_31_complex,
            stateB=ejm_50_complex,
            mapping=edge,
            protocol=complex_rbfe_protocol, 
            name=f"{ejm_31_complex.name}_{ejm_50_complex.name}_complex"
        )
transformation_solvent = Transformation(
            stateA=ejm_31_solvent,
            stateB=ejm_50_solvent,
            mapping=edge,
            protocol=solvent_rbfe_protocol,
            name=f"{ejm_31_solvent.name}_{ejm_50_solvent.name}_solvent"
        )

complex_dag = transformation_complex.create()
solvent_dag = transformation_solvent.create()

#### Step 6 ####
complex_unit = list(complex_dag.protocol_units)[0]
results = complex_unit.run(dry=True, verbose=True)
import pickle
with open(os.path.join(out_dir, "openfe_tyk2_system.pkl"), "wb") as file:
    pickle.dump(results, file)

#### Step 7 ####
top = results["hybrid_factory"].omm_hybrid_topology
positions = results["hybrid_positions"]
system = results["hybrid_system"]
lines = []
for (a,coor) in zip(top.atoms(), positions):
    if a.name=="H11x" and a.residue.chain.id=="5":
        line = ("H12x,"+a.residue.chain.id+","+a.residue.name+","+
                str(a.residue.index+1)+","+
            str(coor[0].value_in_unit(openmm.unit.nanometers))+","+
            str(coor[1].value_in_unit(openmm.unit.nanometers))+","+
            str(coor[2].value_in_unit(openmm.unit.nanometers))+"\n")
    else:
        line = (a.name+","+a.residue.chain.id+","+a.residue.name+","+
                str(a.residue.index+1)+","+
            str(coor[0].value_in_unit(openmm.unit.nanometers))+","+
            str(coor[1].value_in_unit(openmm.unit.nanometers))+","+
            str(coor[2].value_in_unit(openmm.unit.nanometers))+"\n")
    lines.append(line)

with open(os.path.join(out_dir, "positions_openmm.txt"), "w") as f:
    f.writelines(lines)

#### Step 8 ####
λ = 0.0
e,f = calc_energies_and_forces(system, positions, λ)
for key in output_prefix.keys():
    total = 0.0
    for sub in output_prefix[key]:
        total += e[sub].value_in_unit(e[sub].unit)
    with open(os.path.join(out_dir, f"energy_openfe_{key}_l{int(λ)}.txt"), "w") as of:
            of.write(f"{total}\n")

for key in output_prefix.keys():
    total_forces = np.zeros_like(f["NonbondedForce"])
    for sub in output_prefix[key]:
        total_forces += f[sub].value_in_unit(f[sub].unit)
    with open(os.path.join(out_dir, f"forces_openfe_{key}_l{int(λ)}.txt"), "w") as of:
        for force in total_forces:
            of.write(f"{force[0]} {force[1]} {force[2]}\n")

#### Step 9 ####
λ = 1.0
e,f = calc_energies_and_forces(system, positions, λ)
for key in output_prefix.keys():
    total = 0.0
    for sub in output_prefix[key]:
        total += e[sub].value_in_unit(e[sub].unit)
    with open(os.path.join(out_dir, f"energy_openfe_{key}_l{int(λ)}.txt"), "w") as of:
            of.write(f"{total}\n")

for key in output_prefix.keys():
    total_forces = np.zeros_like(f["NonbondedForce"])
    for sub in output_prefix[key]:
        total_forces += f[sub].value_in_unit(f[sub].unit)
    with open(os.path.join(out_dir, f"forces_openfe_{key}_l{int(λ)}.txt"), "w") as of:
        for force in total_forces:
            of.write(f"{force[0]} {force[1]} {force[2]}\n")

#### Step 10 ####
λ = 0.5
e,f = calc_energies_and_forces(system, positions, λ)
for key in output_prefix.keys():
    total = 0.0
    for sub in output_prefix[key]:
        total += e[sub].value_in_unit(e[sub].unit)
    with open(os.path.join(out_dir, f"energy_openfe_{key}_l5.txt"), "w") as of:
            of.write(f"{total}\n")

for key in output_prefix.keys():
    total_forces = np.zeros_like(f["NonbondedForce"])
    for sub in output_prefix[key]:
        total_forces += f[sub].value_in_unit(f[sub].unit)
    with open(os.path.join(out_dir, f"forces_openfe_{key}_l5.txt"), "w") as of:
        for force in total_forces:
            of.write(f"{force[0]} {force[1]} {force[2]}\n")

#### Step 10 ####
λ = 0.25
e,f = calc_energies_and_forces(system, positions, λ)
for key in output_prefix.keys():
    total = 0.0
    for sub in output_prefix[key]:
        total += e[sub].value_in_unit(e[sub].unit)
    with open(os.path.join(out_dir, f"energy_openfe_{key}_l25.txt"), "w") as of:
            of.write(f"{total}\n")

for key in output_prefix.keys():
    total_forces = np.zeros_like(f["NonbondedForce"])
    for sub in output_prefix[key]:
        total_forces += f[sub].value_in_unit(f[sub].unit)
    with open(os.path.join(out_dir, f"forces_openfe_{key}_l25.txt"), "w") as of:
        for force in total_forces:
            of.write(f"{force[0]} {force[1]} {force[2]}\n")

#### Step 10 ####
λ = 0.75
e,f = calc_energies_and_forces(system, positions, λ)
for key in output_prefix.keys():
    total = 0.0
    for sub in output_prefix[key]:
        total += e[sub].value_in_unit(e[sub].unit)
    with open(os.path.join(out_dir, f"energy_openfe_{key}_l75.txt"), "w") as of:
            of.write(f"{total}\n")

for key in output_prefix.keys():
    total_forces = np.zeros_like(f["NonbondedForce"])
    for sub in output_prefix[key]:
        total_forces += f[sub].value_in_unit(f[sub].unit)
    with open(os.path.join(out_dir, f"forces_openfe_{key}_l75.txt"), "w") as of:
        for force in total_forces:
            of.write(f"{force[0]} {force[1]} {force[2]}\n")