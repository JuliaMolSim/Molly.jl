using Molly
using Test
using Lux
using Random
using StaticArrays
using Zygote

@testset "Machine Learning Potentials" begin
    # Setting up a dummy 3-atom system
    coords = [SVector(1.0, 2.0, -1.0), SVector(0.5, 0.0, 1.1), SVector(0.0, -0.5, 0.2)]

    velocities = [SVector(0.0, 0.0, 0.0) for _ in 1:3]

    atoms = [Atom(mass=1.0, σ=0.3, ϵ=0.2), Atom(mass=1.0, σ=0.3, ϵ=0.2), Atom(mass=1.0, σ=0.3, ϵ=0.2)]
    boundary = CubicBoundary(10.0, 10.0, 10.0)
    
    # Setting up the Lux MLP
    rng = Random.default_rng()
    model = Chain(Dense(9 => 16, tanh), Dense(16 => 1))
    ps, st = Lux.setup(rng, model)
    inter = MachineLearningPotential(model, ps, st)
    
    # Setting up the Molly System
    sys = System(
        atoms=atoms,
        coords=coords,
        velocities=velocities, # Add unitless velocities
        boundary=boundary,
        general_inters=(inter,),
        energy_units=NoUnits,  # Force Molly into unitless mode
        force_units=NoUnits    # Force Molly into unitless mode
    )
    
    # Testing Energy Evaluation
    E = Molly.potential_energy(sys, inter)
    @test E isa Number
    @test !isnan(E)
    
    # Testing in Force Evaluation
    F = Molly.forces(sys, inter)
    @test length(F) == 3
    @test F[1] isa SVector{3}
    @test !isnan(F[1][1])
end