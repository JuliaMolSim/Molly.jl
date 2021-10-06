using Molly

using Base.Threads
using Statistics
using Test

@testset "Lennard-Jones gas energy conservation" begin
    temp = 1.0u"K"
    timestep = 0.005u"ps"
    n_steps = 10_000
    box_size = SVector(50.0, 50.0, 50.0)u"nm"
    n_atoms = 2_000
    mass = 40.0u"u"

    parallel_list = nthreads() > 1 ? (false, true) : (false,)
    lj_potentials = (
        LennardJones(cutoff=ShiftedPotentialCutoff(3.0u"nm"), nl_only=false, skip_shortcut=false),
        LennardJones(cutoff=ShiftedPotentialCutoff(3.0u"nm"), nl_only=false, skip_shortcut=true ),
        LennardJones(cutoff=ShiftedForceCutoff(    3.0u"nm"), nl_only=false, skip_shortcut=false),
        LennardJones(cutoff=ShiftedForceCutoff(    3.0u"nm"), nl_only=false, skip_shortcut=true ),
    )

    for parallel in parallel_list
        @testset "$lj_potential" for lj_potential in lj_potentials
            s = Simulation(
                simulator=VelocityVerlet(),
                atoms=[Atom(charge=0.0u"q", mass=mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
                general_inters=(lj_potential,),
                coords=placeatoms(n_atoms, box_size, 0.6u"nm"),
                velocities=[velocity(mass, temp) for i in 1:n_atoms],
                temperature=temp,
                box_size=box_size,
                loggers=Dict("coords" => CoordinateLogger(100),
                                "energy" => EnergyLogger(100)),
                timestep=timestep,
                n_steps=n_steps,
            )

            E0 = energy(s)
            @time simulate!(s; parallel=parallel)

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
