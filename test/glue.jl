using Molly
using Test
using Crystal

fs_inter, elements, masses, bcc_lattice_constants, reference_energies = Molly.get_finnissinclair1984(true)

function forces_are_zero(forces; dims=3)
    return all([isapprox(f, zero(rand(3)), atol=1e-4) for f in forces])
end

function glue_remains_reasonable(glue_init,glue_end)
    return all(isapprox.(glue_init,glue_end,rtol=.1))
end

function groundstate_energy_as_expected(s, element, inter)
    ref = reference_energies[element]
    return isapprox(s.loggers["energy"].energies[1]/length(s.coords), -ref["u"], atol=1e-2)
end

function vacancy_formation_energy_as_expected(s, s_vac, element, inter)
    ref = reference_energies[element]
    u_gs = s.loggers["energy"].energies[1]
    n_atoms = length(s.coords)
    u_vac = s_vac.loggers["energy"].energies[1]
    n_atoms_vac = length(s_vac.coords)
    u_vac_form = u_vac - n_atoms_vac/n_atoms * u_gs
    return isapprox(u_vac_form, ref["u_vac"], atol=7e-2)
end

function run_bcc(element::String, fs_inter, T::Real=.01; 
        nx::Integer=3, ny::Integer=3, nz::Integer=3, vac::Bool=false)

    masses = Dict("V" => 50.9415, "Nb" => 92.9064, "Ta" => 180.9479,
        "Cr" => 51.996, "Mo" => 95.94, "W" => 183.85,
        "Fe" => 55.847)
    a = bcc_lattice_constants[element]
    elements = [element for _ in 1:2]
    el2atom_map = Dict(element => Atom(name=element, mass=masses[element]))
    atoms, coords, box, box_size, box_vectors = Crystal.make_bcc_unitcell(elements, a=a, el2atom_map=el2atom_map)
    sc_atoms, sc_coords, sc_box, sc_box_size = Crystal.make_supercell(atoms, coords, 
        box, box_size, nx=nx, ny=ny, nz=nz)
    
    if vac
        # introducing a vacancy
        sc_atoms, sc_coords = Crystal.add_vacancies(sc_atoms, sc_coords, ixs=[1])
    end
    n_atoms = length(sc_atoms)
    
    velocities = [velocity(sc_atoms[i].mass, T*fs_inter.kb, dims=3) for i in 1:n_atoms]
    nb_matrix = trues(n_atoms,n_atoms)
    dist_cutoff = 2 * a

    loggers = Dict(
        "glue" => GlueDensityLogger(1),
        "forces" => ForceLogger(5),
        "energy" => EnergyLogger(1)
    )

    nf = DistanceNeighbourFinder(nb_matrix, 1, dist_cutoff)
    
    s = Simulation(
        simulator=VelocityVerlet(), 
        atoms=sc_atoms, 
        general_inters=(fs_inter,),
        coords=[SVector{3}(v) for v in sc_coords], 
        velocities=velocities,
        temperature=T*fs_inter.kb, 
        box_size=sc_box_size[1,1],
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
    for element in elements
        @testset "$element" begin
            s = run_bcc(element, fs_inter, T)
            @test forces_are_zero(s.loggers["forces"].forces[1])
            @test glue_remains_reasonable(s.loggers["glue"].glue_densities[1],
                s.loggers["glue"].glue_densities[end])
            @test groundstate_energy_as_expected(s, element, fs_inter)
            
            s_vac = run_bcc(element, fs_inter, T; vac=true)
            @test !forces_are_zero(s_vac.loggers["forces"].forces[1])
            @test vacancy_formation_energy_as_expected(s, s_vac, element, fs_inter)
        end
    end
    
end
