from openmm.app import *
from openmm import *
from openmm.unit import *

# Calculate forces with Gromacs and OpenMM for comparison to Molly using a99SB-disp force field
# Used OpenMM v8.4.0, Python v3.12.12 and Gromacs 2021.4

# protein.gro and topol.top files in Molly/data/a99SB-disp_refs/gromacs_files were 
# prepared with Gromacs' pdb2gmx and editconf using structures in Molly/data/openmm_refs:
# gmx pdb2gmx -f protein.pdb -o processed.gro -p topol.top -ignh -ff 'a99SBdisp'
# gmx editconf -f processed.gro -o protein.gro -c -d 1.0 -bt cubic

# a99SBdisp force field files compatible with Gromacs are required to run this script
# and can be found at https://github.com/paulrobustelli/Force-Fields/tree/master/Gromacs_FFs/a99SBdisp.ff
# a path to the Gromacs force field files must be specified in all topol.top files  

def write_forces_to_file(protein):

    # load files prepared and parameterised with gromacs
    gro = GromacsGroFile(
        f'../data/a99SB-disp_refs/gromacs_files/{protein}/protein.gro'
    )
    top = GromacsTopFile(
        f'../data/a99SB-disp_refs/gromacs_files/{protein}/topol.top', # make sure to add path to Gromacs force field files in topol.top
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir='/Gromacs_FFs/a99SBdisp.ff' # make sure to add path to Gromacs force field files here
    )
    gmx_positions=gro.positions
    
    # setup system
    gmx_system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer, rigidWater=False)
    
    # use reference platform  
    platform = Platform.getPlatformByName('Reference')
    
    # create context
    integrator = VerletIntegrator(0.001*picoseconds) 
    gmx_context = Context(gmx_system, integrator, platform)
    gmx_context.setPositions(gmx_positions)
    
    # get forces per atom
    gmx_state = gmx_context.getState(forces=True)
    gmx_forces = gmx_state.getForces(asNumpy=True)

    # write file with forces
    unit='kJ/(nm mol)'
    with open(f'../data/a99SB-disp_refs/{protein}.dat', 'w') as f:
        for row in gmx_forces:
            f.write(f'{row[0]._value} {unit},{row[1]._value} {unit},{row[2]._value} {unit}\n')

    # write pdb
    with open(f'../data/a99SB-disp_refs/{protein}.pdb', 'w') as f:
        PDBFile.writeFile(top.topology, gro.positions, f)

if __name__=="__main__":

    test_proteins_list = [
        'a-synuclein_1','barn_bar','bpti','cd2_cd58','cole7_im7','drkN_SH3_1','gb3','hewl','NTail_1','PaaA2_1','sgpb_omtky3','ubiquitin','5AWL_A_noHET'
    ]

    for protein in test_proteins_list:
        write_forces_to_file(protein)