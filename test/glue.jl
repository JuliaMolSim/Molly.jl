using Molly
using Test
using Crystal
using DataFrames

elements = ["V", "Nb", "Ta", "Cr", "Mo", "W", "Fe"]
element_pairings = [string(el,el) for el in elements]
element_pair_map = Dict(pair => i for (i,pair) in enumerate(element_pairings))

df = DataFrame(
    element_pair = element_pairings,
    d = [3.692767, 3.915354, 4.076980, 3.915720, 4.114825, 4.400224, 3.699579],
    A = [2.010637, 3.013789, 2.591061, 1.453418, 1.887117, 1.896373, 1.889846],
    β = [0, 0, 0, 1.8, 0, 0, 1.8],
    c = [3.8, 4.2, 4.2, 2.9, 3.25, 3.25, 3.4],
    c₀ = [-0.8816318, -1.5640104, 1.2157373, 29.1429813, 43.4475218, 47.1346499, 1.2110601],
    c₁ = [1.4907756, 2.0055779, 0.0271471, -23.3975027, -31.9332978, -33.7665655, -0.7510840],
    c₂ = [-0.3976370, -0.4663764, -0.1217350, 4.7578297, 6.0804249, 6.2541999, 0.1380773],
)

masses = Dict("V" => 50.9415, "Nb" => 92.9064, "Ta" => 180.9479,
              "Cr" => 51.996, "Mo" => 95.94, "W" => 183.85,
              "Fe" => 55.847)

# Å
bcc_lattice_constants = Dict(
    "V" => 3.0399, "Nb" => 3.3008, 
    "Ta" => 3.3058, "Cr" => 2.8845, "Mo" => 3.1472, 
    "W" => 3.1652, "Fe" => 2.8665
)

reference_energies = DataFrame(
    element_pair = element_pairings,
    u = [5.31, 7.57, 8.1, 4.1, 6.82, 8.9, 4.28],
    u_vac = [1.92, 2.64, 3.13, 1.97, 2.58, 3.71, 1.77]
)

fs_inter = FinnisSinclair(true, element_pair_map, df)

function test_forces_zero(forces, n_atoms; dims=3)
    zeros = [zero(rand(3)) for _ in 1:n_atoms]
    return all(isapprox.(forces, zeros, atol=1e-4))
end

function test_forces_for_element(element::String, fs_inter; nx::Integer=3, ny::Integer=3, nz::Integer=3, )
    a = bcc_lattice_constants[element]
    atoms, coords, box, box_size, box_vectors = Crystal.make_bcc_unitcell(element, a=a)
    sc_atoms, sc_coords, sc_box, sc_box_size = Crystal.make_supercell(atoms, coords, box, box_size, nx=nx, ny=ny, nz=nz)
    n_atoms = length(sc_atoms)
    
    specific_inter_list = ((fs_inter,),)
    velocities = [velocity(1., .01, dims=3) for i in 1:n_atoms]
    sim = VelocityVerlet()
    nb_matrix = trues(n_atoms,n_atoms)
    n_steps = 1
    dist_cutoff = 2 * a

    nf = DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff)

    loggers = Dict("temperature" => TemperatureLogger(1))
    
    s = Simulation(
        simulator=sim, 
        atoms=sc_atoms, 
        specific_inter_lists=specific_inter_list,
        general_inters=(),
        coords=[SVector{3}(v) for v in sc_coords], 
        velocities=velocities,
        temperature=.01, 
        box_size=sc_box_size[1,1],
        timestep=.002,
        n_steps=1,
        neighbour_finder=nf,
        loggers=loggers,
    )
    find_neighbours!(s, s.neighbour_finder, 0)
    forces = accelerations(s, parallel=false)
    return test_forces_zero(forces, n_atoms)
end

function test_bcc_vacancy_forces(fs_inter;element::String="Fe",nx::Int64=3,ny::Int64=3,nz::Int64=3,)
    a = bcc_lattice_constants[element]
    atoms, coords, box, box_size, box_vectors = Crystal.make_bcc_unitcell(element, a=a)
    sc_atoms, sc_coords, sc_box, sc_box_size = Crystal.make_supercell(atoms, coords, box, box_size, nx=nx, ny=ny, nz=nz)
    n_atoms = length(sc_atoms)
            
    # introducing a vacancy
    sc_atoms_vac, sc_coords_vac = Crystal.add_vacancies(sc_atoms, sc_coords, ixs=[1])
    n_atoms_vac = length(sc_atoms_vac)
    
    specific_inter_list = ((fs_inter,),)
    velocities = [velocity(1., .01, dims=3) for i in 1:n_atoms_vac]
    sim = VelocityVerlet()
    nb_matrix = trues(n_atoms_vac,n_atoms_vac)
    dist_cutoff = 2 * a
    n_steps = 1
    nf = DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff)

    loggers = Dict("temperature" => TemperatureLogger(1))

    s = Simulation(
        simulator=sim, 
        atoms=sc_atoms_vac, 
        specific_inter_lists=specific_inter_list,
        general_inters=(),
        coords=[SVector{3}(v) for v in sc_coords_vac], 
        velocities=velocities,
        temperature=.01, 
        box_size=sc_box_size[1,1],
        timestep=.002,
        n_steps=1,
        neighbour_finder=nf,
        loggers=loggers,
    )
    find_neighbours!(s, s.neighbour_finder, 0)
    forces = accelerations(s, parallel=false)
    return !test_forces_zero(forces, n_atoms_vac)
end

function test_energies_for_element(element::String, fs_inter, u; nx::Integer=3, ny::Integer=3, nz::Integer=3, )
    a = bcc_lattice_constants[element]
    atoms, coords, box, box_size, box_vectors = Crystal.make_bcc_unitcell(element, a=a)
    sc_atoms, sc_coords, sc_box, sc_box_size = Crystal.make_supercell(atoms, coords, box, box_size, nx=nx, ny=ny, nz=nz)
    n_atoms = length(sc_atoms)
    
    specific_inter_list = ((fs_inter,),)
    velocities = [velocity(1., .01, dims=3) for i in 1:n_atoms]
    sim = VelocityVerlet()
    nb_matrix = trues(n_atoms,n_atoms)
    n_steps = 1
    dist_cutoff = 2 * a

    nf = DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff);

    loggers = Dict("temperature" => TemperatureLogger(1))
    
    s = Simulation(
        simulator=sim, 
        atoms=sc_atoms, 
        specific_inter_lists=specific_inter_list,
        general_inters=(),
        coords=[SVector{3}(v) for v in sc_coords], 
        velocities=velocities,
        temperature=.01, 
        box_size=sc_box_size[1,1],
        timestep=.002,
        n_steps=1,
        neighbour_finder=nf,
        loggers=loggers,
    )
    find_neighbours!(s, s.neighbour_finder, 0)
    u_md = Molly.potential_energy(fs_inter, s)/n_atoms
    return isapprox(u_md, u, atol=1e-2)
end

function bcc_vacancy_formation_energy(fs_inter;element::String="Fe",nx::Int64=3,ny::Int64=3,nz::Int64=3,)
    a = bcc_lattice_constants[element]
    atoms, coords, box, box_size, box_vectors = Crystal.make_bcc_unitcell(element, a=a)
    sc_atoms, sc_coords, sc_box, sc_box_size = Crystal.make_supercell(atoms, coords, box, box_size, nx=nx, ny=ny, nz=nz)
    n_atoms = length(sc_atoms)
        
    # energy of the vacancy free system
    specific_inter_list = ((fs_inter,),)
    velocities = [velocity(1., .01, dims=3) for i in 1:n_atoms]
    sim = VelocityVerlet()
    nb_matrix = trues(n_atoms,n_atoms)
    n_steps = 1
    dist_cutoff = 2 * a

    nf = DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff);

    loggers = Dict("temperature" => TemperatureLogger(1))

    s = Simulation(
        simulator=sim, 
        atoms=sc_atoms, 
        specific_inter_lists=specific_inter_list,
        general_inters=(),
        coords=[SVector{3}(v) for v in sc_coords], 
        velocities=velocities,
        temperature=.01, 
        box_size=sc_box_size[1,1],
        timestep=.002,
        n_steps=1,
        neighbour_finder=nf,
        loggers=loggers,
    )
    find_neighbours!(s, s.neighbour_finder, 0)
    u_md = Molly.potential_energy(fs_inter, s)
    
    # introducing a vacancy
    sc_atoms_vac, sc_coords_vac = Crystal.add_vacancies(sc_atoms, sc_coords, ixs=[1])
    n_atoms_vac = length(sc_atoms_vac)
    
    velocities = [velocity(1., .01, dims=3) for i in 1:n_atoms_vac]
    nb_matrix = trues(n_atoms_vac,n_atoms_vac)

    nf = DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff);

    loggers = Dict("temperature" => TemperatureLogger(1))

    s = Simulation(
        simulator=sim, 
        atoms=sc_atoms_vac, 
        specific_inter_lists=specific_inter_list,
        general_inters=(),
        coords=[SVector{3}(v) for v in sc_coords_vac], 
        velocities=velocities,
        temperature=.01, 
        box_size=sc_box_size[1,1],
        timestep=.002,
        n_steps=1,
        neighbour_finder=nf,
        loggers=loggers,
    )
    find_neighbours!(s, s.neighbour_finder, 0)
    u_md_vac = Molly.potential_energy(fs_inter, s)
    return u_md_vac - n_atoms_vac/n_atoms * u_md
end


@testset "Finnis-Sinclair" begin
    
    @testset "test forces" begin
        for element in elements
            @test test_forces_for_element(element, fs_inter)
        end
    end
    
    @testset "bcc + vacancy: test forces" begin
        for element in elements
            @test test_bcc_vacancy_forces(fs_inter, element=element)
        end
    end
    
    @testset "potential energies" begin 
        for element in elements
            element_pair = string(element, element)
            row = reference_energies[fs_inter.element_pair_map[element_pair],:]
            @testset "$element" begin 
                @test test_energies_for_element(element, fs_inter, -row.u) 
            end
        end
    end
    
    @testset "potential energies: bcc vacancy" begin 
        for element in elements
            element_pair = string(element, element)
            row = reference_energies[fs_inter.element_pair_map[element_pair],:]
            u_vac = bcc_vacancy_formation_energy(fs_inter, element=element)
            @testset "$element" begin 
                @test isapprox(u_vac, row.u_vac, atol=7e-2) 
            end
        end
    end
end