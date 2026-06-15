# test/dynamic.jl

using Molly
using Test
using StaticArrays

@testset "Dynamic System Resizing" begin
    # Setting up a basic 3-atom system
    coords = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0), SVector(3.0, 3.0, 3.0)]
    velocities = [SVector(0.1, 0.1, 0.1), SVector(0.2, 0.2, 0.2), SVector(0.3, 0.3, 0.3)]
    atoms = [Atom(mass=1.0), Atom(mass=2.0), Atom(mass=3.0)]
    boundary = CubicBoundary(10.0, 10.0, 10.0)
    
    sys = System(
        atoms=atoms, 
        coords=coords, 
        velocities=velocities, 
        boundary=boundary,
        energy_units=NoUnits, 
        force_units=NoUnits
    )
    
    @test length(sys.atoms) == 3
    
    # Testing adding an atom
    new_atom = Atom(mass=4.0)
    new_coord = SVector(4.0, 4.0, 4.0)
    new_vel = SVector(0.4, 0.4, 0.4)
    
    add_atom!(sys, new_atom, new_coord, new_vel)
    
    @test length(sys.atoms) == 4
    @test sys.coords[4] == SVector(4.0, 4.0, 4.0)
    
    # Testing O(1) swap-and-pop removal
    # If we delete atom index 2, atom 4 should take its place
    remove_atom!(sys, 2)
    
    @test length(sys.atoms) == 3
    @test sys.atoms[2].mass == 4.0               # Atom 4 moved to index 2
    @test sys.coords[2] == SVector(4.0, 4.0, 4.0) # Atom 4 coords moved to index 2
end
