using Molly
using Test
using Crystal

function forces_are_zero(forces; dims=3)
    return all([isapprox(f, zero(rand(3)), atol=1e-4) for f in forces])
end

function glue_remains_reasonable(glue_init,glue_end)
    return all(isapprox.(glue_init,glue_end,rtol=.1))
end

function groundstate_energy_as_expected(s, ref)
    return isapprox(s.loggers["energy"].energies[1]/length(s.coords), -ref["u"], atol=1e-2)
end

function vacancy_formation_energy_as_expected(s, s_vac, ref)
    u_gs = s.loggers["energy"].energies[1]
    n_atoms = length(s.coords)
    u_vac = s_vac.loggers["energy"].energies[1]
    n_atoms_vac = length(s_vac.coords)
    u_vac_form = u_vac - n_atoms_vac/n_atoms * u_gs
    return isapprox(u_vac_form, ref["u_vac"], atol=7e-2)
end

function run_bcc(element::String, fs_inter, a::Real; T::Real=.01, 
        nx::Integer=3, ny::Integer=3, nz::Integer=3, vac::Bool=false)

    masses = Dict("V" => 50.9415, "Nb" => 92.9064, "Ta" => 180.9479,
        "Cr" => 51.996, "Mo" => 95.94, "W" => 183.85,
        "Fe" => 55.847)

    elements = [element for _ in 1:2]
    el2atom_map = Dict(element => Atom(name=element, mass=masses[element]))
    unitcell = Crystal.make_bcc_unitcell(elements, a, el2atom_map)
    supercell = Crystal.make_supercell(unitcell, nx=nx, ny=ny, nz=nz)
    
    if vac
        # introducing a vacancy
        supercell = Crystal.add_vacancies(supercell, ixs=[1])
    end
    n_atoms = length(supercell.coords)
    
    velocities = [velocity(atom.mass, T, dims=3) for atom in supercell.atoms]
    nb_matrix = trues(n_atoms,n_atoms)
    dist_cutoff = 2. * a

    loggers = Dict(
        "glue" => GlueDensityLogger(1),
        "forces" => ForceLogger(5),
        "energy" => EnergyLogger(1)
    )

    nf = DistanceNeighbourFinder(nb_matrix, 1, dist_cutoff)
    
    s = Simulation(
        simulator=VelocityVerlet(), 
        atoms=supercell.atoms,  
        general_inters=(fs_inter,),
        coords=[SVector{3}(v) for v in supercell.coords],  
        velocities=velocities,
        temperature=T, 
        box_size=supercell.edge_lengths[1], 
        timestep=.002,
        n_steps=100,
        neighbour_finder=nf,
        loggers=loggers,
    )
    simulate!(s)
    return s
end

@testset "Glue: Finnis-Sinclair" begin
    T = .01 # K
    fs_inter, elements, masses, bcc_lattice_constants, reference_energies = Molly.get_finnissinclair1984(true)
    
    T *= fs_inter.kb
    for element in elements
        @testset "$element" begin
            a = bcc_lattice_constants[element]

            s = run_bcc(element, fs_inter, a, T=T, vac=false)
            @testset "forces zero" begin
                @test forces_are_zero(s.loggers["forces"].forces[1])
            end
            @testset "reasonable glue" begin
                @test glue_remains_reasonable(s.loggers["glue"].glue_densities[1],
                    s.loggers["glue"].glue_densities[end])
            end
            @testset "checking groundstate" begin
                @test groundstate_energy_as_expected(s, reference_energies[element])
            end
            
            s_vac = run_bcc(element, fs_inter, a, T=T, vac=true)
            @testset "vacancy forces nonzero" begin
                @test !forces_are_zero(s_vac.loggers["forces"].forces[1])
            end
            @testset "vacancy formation energy" begin
                @test vacancy_formation_energy_as_expected(s, s_vac, reference_energies[element])
            end
        end
    end
    
end
