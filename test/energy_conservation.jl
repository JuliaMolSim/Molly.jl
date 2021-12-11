using Molly

using Base.Threads
using Statistics
using Test

@testset "Lennard-Jones gas energy conservation" begin
    temp = 1.0u"K"
    n_steps = 10_000
    box_size = SVector(50.0, 50.0, 50.0)u"nm"
    n_atoms = 2_000
    atom_mass = 40.0u"u"
    simulator = VelocityVerlet(dt=0.005u"ps")

    parallel_list = nthreads() > 1 ? (false, true) : (false,)
    lj_potentials = (
        LennardJones(cutoff=DistanceCutoff(        3.0u"nm"), nl_only=false),
        LennardJones(cutoff=ShiftedPotentialCutoff(3.0u"nm"), nl_only=false),
        LennardJones(cutoff=ShiftedForceCutoff(    3.0u"nm"), nl_only=false),
    )

    for parallel in parallel_list
        @testset "$lj_potential" for lj_potential in lj_potentials
            s = System(
                atoms=[Atom(charge=0.0, mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
                general_inters=(lj_potential,),
                coords=place_atoms(n_atoms, box_size, 0.6u"nm"),
                velocities=[velocity(atom_mass, temp) for i in 1:n_atoms],
                box_size=box_size,
                loggers=Dict("coords" => CoordinateLogger(100),
                                "energy" => EnergyLogger(100)),
            )

            E0 = energy(s)
            @time simulate!(s, simulator, n_steps; parallel=parallel)

            ΔE = energy(s) - E0
            @test abs(ΔE) < 2e-2u"kJ * mol^-1"

            Es = s.loggers["energy"].energies
            maxΔE = maximum(abs.(Es .- E0))
            @test maxΔE < 2e-2u"kJ * mol^-1"

            @test abs(Es[end] - Es[1]) < 2e-2u"kJ * mol^-1"

            final_coords = last(s.loggers["coords"].coords)
            @test all(all(c .> 0.0u"nm") for c in final_coords)
            @test all(all(c .< box_size) for c in final_coords)
        end
    end
end
