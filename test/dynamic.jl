# test/dynamic.jl
using Molly
using Test
using StaticArrays

@testset "Dynamic System Resizing" begin
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
    
    # Test adding an atom (OUT-OF-PLACE)
    new_atom = Atom(mass=4.0)
    new_coord = SVector(4.0, 4.0, 4.0)
    new_vel = SVector(0.4, 0.4, 0.4)
    
    sys = add_atom(sys, new_atom, new_coord, new_vel)
    
    @test length(sys.atoms) == 4
    @test sys.coords[4] == SVector(4.0, 4.0, 4.0)
    
    # Test standard ordered deletion (OUT-OF-PLACE)
    sys = remove_atom(sys, 2)
    
    @test length(sys.atoms) == 3
    @test sys.atoms[2].mass == 3.0
    @test sys.coords[2] == SVector(3.0, 3.0, 3.0)
end
